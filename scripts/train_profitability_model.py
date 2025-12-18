import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # --- 1. Load the dataset ---
    print("Loading the feature-engineered dataset...")
    df = pd.read_csv('data/daily_features_for_model.csv')
    df['date'] = pd.to_datetime(df['date'])

    # --- 2. Prepare data for the model ---
    # The 'profitability_bracket' is our target variable (y)
    # All other relevant columns are our features (X)
    X = df.drop(columns=['date', 'perfect_trade_profit_pct', 'profitability_bracket'])
    y = df['profitability_bracket']

    # --- 3. Split data chronologically ---
    # We will train on all data up to the end of 2024, and test on 2025
    print("Splitting data into training (pre-2025) and testing (2025) sets...")
    train_end_date = '2024-12-31'
    X_train = X[df['date'] <= train_end_date]
    y_train = y[df['date'] <= train_end_date]
    X_test = X[df['date'] > train_end_date]
    y_test = y[df['date'] > train_end_date]

    if len(X_test) == 0:
        print("No data available for the testing period (2025). Please check your dataset.")
        return
        
    print(f"Training set size: {len(X_train)} days")
    print(f"Testing set size: {len(X_test)} days")

    # --- 4. Train the LightGBM Classifier ---
    print("\nTraining the LightGBM model...")
    lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgb_clf.fit(X_train, y_train)

    # --- 5. Evaluate the model ---
    print("\nEvaluating the model on the 2025 test data...")
    y_pred = lgb_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on 2025 data: {accuracy:.2f}")
    
    print("\nClassification Report for 2025 data:")
    print(classification_report(y_test, y_pred))

    # --- 6. Save the trained model ---
    model_path = 'models/profitability_bracket_model.joblib'
    joblib.dump(lgb_clf, model_path)
    print(f"\nTrained model saved to {model_path}")

if __name__ == "__main__":
    main()
