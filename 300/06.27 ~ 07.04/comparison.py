import pyupbit
import numpy as np

df2 = pyupbit.get_ohlcv("KRW-BTC", count = 2000)
df2['date'] = df2.index
target_year = 2021
df= df2[df2['date'].dt.year == target_year]

df['range'] = (df['high'] - df['low']) * 0.5 # k = 0.5
df['target'] = df['open'] + df['range'].shift(1)
df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
df['bull'] = df['open'] > df['ma5']

fee =  0.0006
df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
                    df['close'] / df['target'] - fee,
                     1)

df['hpr'] = df['ror'].cumprod()
df['holding'] = df['close'].iloc[-1] / df['close'].iloc[0]
# print("backtest for the last {} days".format(days))
print("HPR: ", df['hpr'].iloc[-1])
df.to_excel("comparision.xlsx")