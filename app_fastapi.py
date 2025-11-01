from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import glob
import os

app = FastAPI()
MODEL = None
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'artifacts', 'rf_model.joblib')
if os.path.exists(model_path):
    MODEL = joblib.load(model_path)
PARQUET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'parquet')

@app.get('/health')
def health():
    return {'status':'ok'}

@app.get('/tickers')
def list_tickers():
    parts = []
    if os.path.isdir(PARQUET_DIR):
        parts = [d for d in os.listdir(PARQUET_DIR) if d.startswith('Ticker=')]
    tickers = [p.split('=')[1] for p in parts]
    return {'tickers': tickers}

@app.get('/predict/{ticker}')
def predict(ticker: str):
    files = glob.glob(os.path.join(PARQUET_DIR, f'Ticker={ticker}', '*.parquet'))
    if not files:
        files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    last = df.sort_values('Date').iloc[-1]
    X = last[['Close','Volume','ma_5','ma_20','ma_50','rsi']].values.reshape(1,-1)
    pred = int(MODEL.predict(X)[0]) if MODEL is not None else None
    return {'ticker': ticker, 'prediction': 'UP' if pred==1 else ('DOWN' if pred==0 else 'MODEL_NOT_AVAILABLE')}

@app.get('/top_stocks')
def top_stocks():
    files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
    if not files:
        return {'top': []}
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    today = df['Date'].max()
    df_today = df[df['Date']==today]
    df_today['pct_change'] = df_today.groupby('Ticker')['Close'].pct_change().fillna(0)
    top = df_today.sort_values('pct_change', ascending=False).head(10)[['Ticker','pct_change']].to_dict(orient='records')
    return {'top': top}

@app.get('/bottom_stocks')
def bottom_stocks():
    files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
    if not files:
        return {'bottom': []}
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    today = df['Date'].max()
    df_today = df[df['Date']==today]
    df_today['pct_change'] = df_today.groupby('Ticker')['Close'].pct_change().fillna(0)
    bottom = df_today.sort_values('pct_change', ascending=True).head(10)[['Ticker','pct_change']].to_dict(orient='records')
    return {'bottom': bottom}

if __name__ == '__main__':
    uvicorn.run('app_fastapi:app', host='0.0.0.0', port=8000, reload=True)
