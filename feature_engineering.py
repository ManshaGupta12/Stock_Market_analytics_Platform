from pyspark.sql import functions as F
from pyspark.sql import Window

def add_moving_average(df, price_col='Close', windows=(5, 20, 50, 200)):
    out = df
    for w in windows:
        win = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-w+1, 0)
        out = out.withColumn(f'ma_{w}', F.avg(F.col(price_col)).over(win))
    return out

def add_rsi(df, period=14):
    win = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-period+1, 0)
    delta = F.col('Close') - F.lag('Close').over(Window.partitionBy('Ticker').orderBy('Date'))
    gain = F.when(delta > 0, delta).otherwise(0.0)
    loss = F.when(delta < 0, -delta).otherwise(0.0)
    avg_gain = F.avg(gain).over(win)
    avg_loss = F.avg(loss).over(win)
    rs = avg_gain / (avg_loss + F.lit(1e-6))
    rsi = 100 - (100 / (1 + rs))
    return df.withColumn('rsi', rsi)

# Note: For EMA/MACD, consider pandas UDF for numerical stability.
