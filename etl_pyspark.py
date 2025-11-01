import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import yfinance as yf
import pandas as pd
from datetime import datetime
from feature_engineering import add_moving_average, add_rsi

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PARQUET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'parquet')

spark = SparkSession.builder.master('local[*]').appName('stock-etl').getOrCreate()

TICKERS_FILE = os.path.join(DATA_DIR, 'sample_tickers.txt')

def fetch_yfinance(ticker, period='2y'):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df['Ticker'] = ticker
    df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def main():
    with open(TICKERS_FILE) as f:
        tickers = [l.strip() for l in f if l.strip()]
    pdfs = []
    for t in tickers:
        print('Fetching', t)
        p = fetch_yfinance(t, period='2y')
        if p is not None:
            pdfs.append(p)
    if not pdfs:
        print('No data')
        return
    big = pd.concat(pdfs, ignore_index=True)
    big['Date'] = pd.to_datetime(big['Date']).dt.date
    big = big.dropna()
    sdf = spark.createDataFrame(big)
    sdf = sdf.withColumn('Date', F.to_date(F.col('Date').cast('string')))
    sdf = add_moving_average(sdf, price_col='Close', windows=(5,20,50,200))
    sdf = add_rsi(sdf, period=14)
    q = sdf.approxQuantile('Volume', [0.99], 0.01)
    cap = q[0] if q else None
    if cap:
        sdf = sdf.withColumn('Volume', F.when(F.col('Volume') > cap, cap).otherwise(F.col('Volume')))
    out_dir = PARQUET_DIR
    sdf.write.mode('overwrite').partitionBy('Ticker').parquet(out_dir)
    print('Wrote parquet to', out_dir)

if __name__ == '__main__':
    main()
