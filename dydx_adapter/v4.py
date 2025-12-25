# Timestamp (UTC): 2025-12-24T13:14:02Z
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
from typing import Any, Dict, Optional

from dydx_v4_client import MAX_CLIENT_ID
from dydx_v4_client.network import TESTNET, make_mainnet, Network
from dydx_v4_client.key_pair import KeyPair
from dydx_v4_client.node.client import NodeClient
from dydx_v4_client.wallet import Wallet
from dydx_v4_client.utility import Usdc

from dydx_v4_client.indexer.rest.indexer_client import IndexerClient

from v4_proto.cosmos.base.query.v1beta1.pagination_pb2 import PageRequest
from v4_proto.dydxprotocol.assets import query_pb2 as assets_query
from v4_proto.dydxprotocol.assets import query_pb2_grpc as assets_query_grpc


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    s = _env_str(name, "")
    if not s:
        return int(default)
    return int(s)


def _env_float(name: str, default: float) -> float:
    s = _env_str(name, "")
    if not s:
        return float(default)
    return float(s)


@dataclass
class DydxV4Config:
    env: str

    mnemonic: str
    address: str

    indexer_rest: str
    indexer_ws: str
    node_grpc: str

    subaccount_trading: int
    subaccount_bank: int

    market: str

    trade_floor_usdc: float
    profit_siphon_frac: float
    fee_side: float

    # Risk / bankroll policy
    # - Floor top-ups from bank->trading are only allowed when bank_equity < bank_threshold_usdc.
    # - If a liquidation is detected and bank_equity >= bank_threshold_usdc, transfer
    #   bank_equity * liquidation_recap_bank_frac from bank->trading.
    bank_threshold_usdc: float
    liquidation_recap_bank_frac: float

    use_margin_frac: float
    leverage_safety_frac: float
    max_leverage: int

    usdc_asset_id_override: Optional[int] = None

    @classmethod
    def from_env(cls) -> "DydxV4Config":
        env = _env_str("DYDX_ENV", "mainnet").lower()

        mnemonic = _env_str("DYDX_MNEMONIC", "")
        address = _env_str("DYDX_ADDRESS", "")

        indexer_rest = _env_str("DYDX_INDEXER_REST", "https://indexer.dydx.trade")
        indexer_ws = _env_str("DYDX_INDEXER_WS", "wss://indexer.dydx.trade/v4/ws")
        node_grpc = _env_str("DYDX_NODE_GRPC", "")

        sub_trading = _env_int("DYDX_SUBACCOUNT_TRADING", 1)
        sub_bank = _env_int("DYDX_SUBACCOUNT_BANK", 0)

        market = _env_str("DYDX_MARKET", "BTC-USD")

        trade_floor_usdc = _env_float("DYDX_TRADE_FLOOR_USDC", 10.0)
        profit_siphon_frac = _env_float("DYDX_PROFIT_SIPHON_FRAC", 0.30)
        fee_side = _env_float("DYDX_FEE_SIDE", 0.001)

        # Bankroll policy
        bank_threshold_usdc = _env_float("DYDX_BANK_THRESHOLD_USDC", 280.0)
        liquidation_recap_bank_frac = _env_float("DYDX_LIQUIDATION_RECAP_BANK_FRAC", 0.10)

        # For max leverage sizing, defaults are 1.0 (use all equity as margin; no safety haircut).
        # You can still set these lower via env to keep extra buffer.
        use_margin_frac = _env_float("DYDX_USE_MARGIN_FRAC", 1.0)
        leverage_safety_frac = _env_float("DYDX_LEVERAGE_SAFETY_FRAC", 1.0)

        # Max leverage cap. If <= 0 (or 'auto'), we will use the market's max leverage.
        raw_max_lev = _env_str("DYDX_MAX_LEVERAGE", "auto").lower()
        if raw_max_lev in {"", "auto", "max"}:
            max_leverage = 0
        else:
            max_leverage = int(raw_max_lev)

        usdc_asset_id_override = None
        raw_asset_id = _env_str("DYDX_USDC_ASSET_ID", "")
        if raw_asset_id:
            try:
                usdc_asset_id_override = int(raw_asset_id)
            except Exception:
                usdc_asset_id_override = None

        return cls(
            env=env,
            mnemonic=mnemonic,
            address=address,
            indexer_rest=indexer_rest,
            indexer_ws=indexer_ws,
            node_grpc=node_grpc,
            subaccount_trading=sub_trading,
            subaccount_bank=sub_bank,
            market=market,
            trade_floor_usdc=float(trade_floor_usdc),
            profit_siphon_frac=float(profit_siphon_frac),
            fee_side=float(fee_side),
            bank_threshold_usdc=float(bank_threshold_usdc),
            liquidation_recap_bank_frac=float(liquidation_recap_bank_frac),
            use_margin_frac=float(use_margin_frac),
            leverage_safety_frac=float(leverage_safety_frac),
            max_leverage=int(max_leverage),
            usdc_asset_id_override=usdc_asset_id_override,
        )

    def make_network(self) -> Network:
        if self.env == "testnet":
            return TESTNET
        if self.env != "mainnet":
            raise ValueError(f"Unsupported DYDX_ENV={self.env!r} (expected mainnet|testnet)")

        if not self.node_grpc:
            raise ValueError("DYDX_NODE_GRPC is required for mainnet")

        return make_mainnet(
            rest_indexer=str(self.indexer_rest).rstrip("/"),
            websocket_indexer=str(self.indexer_ws).rstrip("/"),
            node_url=str(self.node_grpc).strip(),
        )


@dataclass
class DydxV4Clients:
    network: Network
    node: NodeClient
    indexer: IndexerClient
    wallet: Wallet
    usdc_asset_id: int


async def _query_usdc_asset_id(node: NodeClient) -> int:
    stub = assets_query_grpc.QueryStub(node.channel)
    req = assets_query.QueryAllAssetsRequest(pagination=PageRequest(limit=500))
    resp = stub.AllAssets(req)

    assets = list(resp.asset)
    for a in assets:
        try:
            if str(a.symbol).upper() == "USDC":
                return int(a.id)
        except Exception:
            continue

    sym = []
    for a in assets:
        try:
            sym.append(str(a.symbol))
        except Exception:
            continue
    raise RuntimeError(f"Failed to discover USDC asset id (available symbols: {sorted(set(sym))[:50]})")


def usdc_to_quantums(amount_usdc: float) -> int:
    # Use Decimal(str(x)) to avoid float binary rounding surprises.
    usdc = Usdc(Decimal(str(float(amount_usdc))))
    return int(usdc.quantize_as_u64())


def new_client_id() -> int:
    # 32-bit client id.
    import secrets

    return int(secrets.randbelow(MAX_CLIENT_ID))


def _address_from_mnemonic(mnemonic: str) -> str:
    # Derive bech32 "dydx" address from mnemonic (no network calls).
    kp = KeyPair.from_mnemonic(str(mnemonic))
    w = Wallet(key=kp, account_number=0, sequence=0)
    return str(w.address)


async def connect_v4(cfg: DydxV4Config) -> DydxV4Clients:
    if not cfg.mnemonic:
        raise ValueError("DYDX_MNEMONIC is required")

    address = str(cfg.address).strip() or _address_from_mnemonic(cfg.mnemonic)
    cfg.address = address

    network = cfg.make_network()
    node = await NodeClient.connect(network.node)
    wallet = await Wallet.from_mnemonic(node, cfg.mnemonic, address)
    indexer = IndexerClient(network.rest_indexer)

    usdc_asset_id = int(cfg.usdc_asset_id_override) if cfg.usdc_asset_id_override is not None else int(await _query_usdc_asset_id(node))

    return DydxV4Clients(
        network=network,
        node=node,
        indexer=indexer,
        wallet=wallet,
        usdc_asset_id=usdc_asset_id,
    )
