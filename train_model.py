import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PARQUET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'parquet')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'artifacts')
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    import glob
    files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
    if not files:
        print('No parquet files. Run ETL first.')
        return
    df_list = [pd.read_parquet(p) for p in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['Ticker','Date'])
    df['pct_change'] = df.groupby('Ticker')['Close'].pct_change()
    df['target'] = (df.groupby('Ticker')['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = ['Close','Volume','ma_5','ma_20','ma_50','rsi']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy', accuracy_score(y_test, preds))
    joblib.dump(model, os.path.join(MODEL_DIR, 'rf_model.joblib'))
    print('Saved model')

if __name__ == '__main__':
    main()
