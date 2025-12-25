#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-22T15:48:42Z
"""End-of-day report generator.

Reads outputs produced by the live runner in an output directory (default: data/live):
- trades_*.csv
- daily_*.csv
- gate_state.json
- strategy_ledger.json

Writes:
- eod_report_<ts>.html
- eod_summary_<ts>.csv

Optionally, it can email the report via SMTP.
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_matching(dir_path: Path, pattern: str) -> Optional[Path]:
    paths = list(Path(dir_path).glob(pattern))
    if not paths:
        return None
    # Prefer mtime over parsing timestamps.
    return max(paths, key=lambda p: p.stat().st_mtime)


@dataclass
class ReportInputs:
    out_dir: Path
    trades_csv: Optional[Path]
    daily_csv: Optional[Path]
    gate_state_json: Optional[Path]
    ledger_json: Optional[Path]


def _compute_daily_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_trades",
                "mean_daily_pct",
                "sum_daily_pct",
                "median_daily_pct",
                "top_day_pct",
                "worst_day_pct",
            ]
        )

    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce", utc=True)
    t["date"] = t["entry_time"].dt.date
    d = t.groupby("date", as_index=False).agg(
        n_trades=("realized_ret_pct", "size"),
        mean_daily_pct=("realized_ret_pct", "mean"),
        sum_daily_pct=("realized_ret_pct", "sum"),
        median_daily_pct=("realized_ret_pct", "median"),
        top_day_pct=("realized_ret_pct", "max"),
        worst_day_pct=("realized_ret_pct", "min"),
    )
    return d


def build_report(inputs: ReportInputs) -> Tuple[Dict[str, Any], pd.DataFrame]:
    gate_obj = _load_json(inputs.gate_state_json) if inputs.gate_state_json else None
    ledger_obj = _load_json(inputs.ledger_json) if inputs.ledger_json else None

    trades = pd.DataFrame()
    if inputs.trades_csv and inputs.trades_csv.exists():
        trades = pd.read_csv(inputs.trades_csv)

    daily = pd.DataFrame()
    if inputs.daily_csv and inputs.daily_csv.exists():
        daily = pd.read_csv(inputs.daily_csv)
    elif not trades.empty:
        daily = _compute_daily_from_trades(trades)
    else:
        daily = _compute_daily_from_trades(pd.DataFrame(columns=["entry_time", "realized_ret_pct"]))

    gate_enabled = bool(gate_obj.get("enabled")) if isinstance(gate_obj, dict) else False
    gate_window = int(gate_obj.get("window", 0)) if isinstance(gate_obj, dict) else 0
    outcomes = list(gate_obj.get("outcomes", [])) if isinstance(gate_obj, dict) else []
    wins = int(sum(1 for x in outcomes if bool(x)))
    losses = int(len(outcomes) - wins)

    ledger_cap = float(ledger_obj.get("trading_capital_eur", 0.0)) if isinstance(ledger_obj, dict) else 0.0
    ledger_bank = float(ledger_obj.get("bank_eur", 0.0)) if isinstance(ledger_obj, dict) else 0.0

    summary: Dict[str, Any] = {
        "report_created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(inputs.out_dir),
        "trades_csv": str(inputs.trades_csv) if inputs.trades_csv else "",
        "daily_csv": str(inputs.daily_csv) if inputs.daily_csv else "",
        "gate_state_json": str(inputs.gate_state_json) if inputs.gate_state_json else "",
        "strategy_ledger_json": str(inputs.ledger_json) if inputs.ledger_json else "",
        "gate_enabled": gate_enabled,
        "gate_enabled_utc": (gate_obj.get("enabled_utc") if isinstance(gate_obj, dict) else None),
        "gate_window": gate_window,
        "gate_outcomes": json.dumps(outcomes, ensure_ascii=False),
        "gate_wins": wins,
        "gate_losses": losses,
        "ledger_trading_capital_eur": ledger_cap,
        "ledger_bank_eur": ledger_bank,
        "ledger_equity_eur": ledger_cap + ledger_bank,
        "ledger_updated_utc": (ledger_obj.get("updated_utc") if isinstance(ledger_obj, dict) else None),
        "ledger_n_trades": (int(ledger_obj.get("n_trades", 0)) if isinstance(ledger_obj, dict) else 0),
        "ledger_last_trade_pnl_eur": (ledger_obj.get("last_trade_pnl_eur") if isinstance(ledger_obj, dict) else None),
        "ledger_last_trade_realized_ret_pct": (
            ledger_obj.get("last_trade_realized_ret_pct") if isinstance(ledger_obj, dict) else None
        ),
        "ledger_last_trade_leverage": (ledger_obj.get("last_trade_leverage") if isinstance(ledger_obj, dict) else None),
    }

    if not trades.empty and "entry_time" in trades.columns:
        et = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True)
        summary["trades_first_entry_utc"] = et.min().isoformat() if et.notna().any() else None
        summary["trades_last_entry_utc"] = et.max().isoformat() if et.notna().any() else None
        summary["trades_n_rows"] = int(len(trades))
    else:
        summary["trades_first_entry_utc"] = None
        summary["trades_last_entry_utc"] = None
        summary["trades_n_rows"] = 0

    return summary, daily


def _split_recipients(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").replace(";", ",").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _send_email_smtp(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    use_starttls: bool,
    use_ssl: bool,
    mail_from: str,
    mail_to: List[str],
    subject: str,
    body_text: str,
    body_html: Optional[str],
    attachments: List[Path],
) -> None:
    if not host:
        raise SystemExit("SMTP host is required")
    if not mail_from:
        raise SystemExit("Email from address is required")
    if not mail_to:
        raise SystemExit("Email to address(es) are required")

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg["Subject"] = subject
    msg.set_content(body_text)
    if body_html is not None:
        msg.add_alternative(body_html, subtype="html")

    for p in attachments:
        data = p.read_bytes()
        name = p.name
        if name.endswith(".html"):
            maintype, subtype = "text", "html"
        elif name.endswith(".csv"):
            maintype, subtype = "text", "csv"
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, int(port), timeout=20, context=context) as smtp:
            if username:
                smtp.login(username, password)
            smtp.send_message(msg)
        return

    with smtplib.SMTP(host, int(port), timeout=20) as smtp:
        smtp.ehlo()
        if use_starttls:
            context = ssl.create_default_context()
            smtp.starttls(context=context)
            smtp.ehlo()
        if username:
            smtp.login(username, password)
        smtp.send_message(msg)


def write_report(*, inputs: ReportInputs, html_path: Path, csv_path: Path) -> None:
    summary, daily = build_report(inputs)

    # One-row CSV summary.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(csv_path, index=False)

    # Simple HTML.
    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    daily_html = daily.to_html(index=False, escape=True)

    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'/>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1'/>",
            "  <title>EOD Report</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 16px; }",
            "    code { background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }",
            "    th { background: #f6f8fa; text-align: left; }",
            "  </style>",
            "</head>",
            "<body>",
            f"<h1>EOD Report</h1>",
            f"<p><b>Created (UTC):</b> <code>{esc(summary.get('report_created_utc'))}</code></p>",
            "<h2>Inputs</h2>",
            "<ul>",
            f"  <li>out_dir: <code>{esc(summary.get('out_dir'))}</code></li>",
            f"  <li>trades_csv: <code>{esc(summary.get('trades_csv'))}</code></li>",
            f"  <li>daily_csv: <code>{esc(summary.get('daily_csv'))}</code></li>",
            f"  <li>gate_state_json: <code>{esc(summary.get('gate_state_json'))}</code></li>",
            f"  <li>strategy_ledger_json: <code>{esc(summary.get('strategy_ledger_json'))}</code></li>",
            "</ul>",
            "<h2>Gate</h2>",
            "<ul>",
            f"  <li>enabled: <code>{esc(summary.get('gate_enabled'))}</code></li>",
            f"  <li>enabled_utc: <code>{esc(summary.get('gate_enabled_utc'))}</code></li>",
            f"  <li>window: <code>{esc(summary.get('gate_window'))}</code></li>",
            f"  <li>wins/losses: <code>{esc(summary.get('gate_wins'))}</code> / <code>{esc(summary.get('gate_losses'))}</code></li>",
            f"  <li>outcomes: <code>{esc(summary.get('gate_outcomes'))}</code></li>",
            "</ul>",
            "<h2>Ledger</h2>",
            "<ul>",
            f"  <li>trading_capital_eur: <code>{esc(summary.get('ledger_trading_capital_eur'))}</code></li>",
            f"  <li>bank_eur: <code>{esc(summary.get('ledger_bank_eur'))}</code></li>",
            f"  <li>equity_eur: <code>{esc(summary.get('ledger_equity_eur'))}</code></li>",
            f"  <li>updated_utc: <code>{esc(summary.get('ledger_updated_utc'))}</code></li>",
            f"  <li>n_trades: <code>{esc(summary.get('ledger_n_trades'))}</code></li>",
            f"  <li>last_trade_pnl_eur: <code>{esc(summary.get('ledger_last_trade_pnl_eur'))}</code></li>",
            f"  <li>last_trade_realized_ret_pct: <code>{esc(summary.get('ledger_last_trade_realized_ret_pct'))}</code></li>",
            f"  <li>last_trade_leverage: <code>{esc(summary.get('ledger_last_trade_leverage'))}</code></li>",
            "</ul>",
            "<h2>Daily performance</h2>",
            daily_html,
            "</body>",
            "</html>",
        ]
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate EOD HTML+CSV report from live runner outputs")
    ap.add_argument("--out-dir", type=str, default="data/live")
    ap.add_argument("--trades-csv", type=str, default="")
    ap.add_argument("--daily-csv", type=str, default="")
    ap.add_argument("--gate-state", type=str, default="")
    ap.add_argument("--ledger", type=str, default="")
    ap.add_argument("--html-out", type=str, default="")
    ap.add_argument("--csv-out", type=str, default="")

    # Email (optional). Prefer env vars to avoid secrets on the CLI.
    ap.add_argument("--send-email", action="store_true", help="If set, email the report via SMTP")
    ap.add_argument("--email-to", type=str, default="", help="Comma-separated recipients (or set EMAIL_TO)")
    ap.add_argument("--email-from", type=str, default="", help="From address (or set EMAIL_FROM)")
    ap.add_argument("--email-subject", type=str, default="", help="Subject override")

    ap.add_argument("--smtp-host", type=str, default="", help="SMTP host (or set SMTP_HOST)")
    ap.add_argument("--smtp-port", type=int, default=0, help="SMTP port (or set SMTP_PORT)")
    ap.add_argument("--smtp-user", type=str, default="", help="SMTP username (or set SMTP_USER)")
    ap.add_argument("--smtp-pass", type=str, default="", help="SMTP password/app-password (or set SMTP_PASS)")
    ap.add_argument("--smtp-starttls", action="store_true", default=False, help="Use STARTTLS (or set SMTP_STARTTLS=1)")
    ap.add_argument("--smtp-ssl", action="store_true", default=False, help="Use SMTP over SSL (or set SMTP_SSL=1)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    trades_csv = Path(args.trades_csv) if args.trades_csv else _latest_matching(out_dir, "trades_*.csv")
    daily_csv = Path(args.daily_csv) if args.daily_csv else _latest_matching(out_dir, "daily_*.csv")

    gate_state_json = Path(args.gate_state) if args.gate_state else (out_dir / "gate_state.json")
    ledger_json = Path(args.ledger) if args.ledger else (out_dir / "strategy_ledger.json")

    inputs = ReportInputs(
        out_dir=out_dir,
        trades_csv=trades_csv,
        daily_csv=daily_csv,
        gate_state_json=(gate_state_json if gate_state_json.exists() else None),
        ledger_json=(ledger_json if ledger_json.exists() else None),
    )

    ts = _now_ts()
    html_path = Path(args.html_out) if args.html_out else (out_dir / f"eod_report_{ts}.html")
    csv_path = Path(args.csv_out) if args.csv_out else (out_dir / f"eod_summary_{ts}.csv")

    write_report(inputs=inputs, html_path=html_path, csv_path=csv_path)

    print("Wrote:")
    print("  HTML:", html_path)
    print("  CSV :", csv_path)

    if args.send_email:
        smtp_host = str(args.smtp_host or os.getenv("SMTP_HOST", ""))
        smtp_port = int(args.smtp_port or int(os.getenv("SMTP_PORT", "0") or 0) or 587)
        smtp_user = str(args.smtp_user or os.getenv("SMTP_USER", ""))
        smtp_pass = str(args.smtp_pass or os.getenv("SMTP_PASS", ""))

        env_starttls = str(os.getenv("SMTP_STARTTLS", "")).strip().lower() in {"1", "true", "yes"}
        env_ssl = str(os.getenv("SMTP_SSL", "")).strip().lower() in {"1", "true", "yes"}

        use_starttls = bool(args.smtp_starttls or env_starttls or (not args.smtp_ssl and not env_ssl))
        use_ssl = bool(args.smtp_ssl or env_ssl)

        email_from = str(args.email_from or os.getenv("EMAIL_FROM", ""))
        email_to = _split_recipients(str(args.email_to or os.getenv("EMAIL_TO", "")))

        subj = str(args.email_subject).strip()
        if not subj:
            subj = f"EOD Report ({ts} UTC)"

        body_text = "EOD report attached (HTML + CSV)."
        body_html = html_path.read_text(encoding="utf-8")

        _send_email_smtp(
            host=smtp_host,
            port=int(smtp_port),
            username=smtp_user,
            password=smtp_pass,
            use_starttls=bool(use_starttls and not use_ssl),
            use_ssl=bool(use_ssl),
            mail_from=email_from,
            mail_to=email_to,
            subject=subj,
            body_text=body_text,
            body_html=body_html,
            attachments=[csv_path, html_path],
        )
        print("Email: sent")


if __name__ == "__main__":
    main()
