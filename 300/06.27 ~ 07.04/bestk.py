import pyupbit
import numpy as np

# best hpr을 도출하는 k값 찾기

days = 10

def get_hpr(k=0.5):
    df = pyupbit.get_ohlcv("KRW-BTC", count = days)
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)

    fee =  0.0006
    df['ror'] = np.where(df['high'] > df['target'],
                         df['close'] / df['target'] - fee,
                         1)

    # ror = df['ror'].cumprod()[-2]
    df['hpr'] = df['ror'].cumprod()
    df.to_excel("bestk.xlsx")
    return df['hpr'].iloc[-1]

for k in np.arange(0.1, 1.0, 0.1):
    hpr = get_hpr(k)
    # print("%f %.2f %f" % (days, k, hpr))
    print('for the last {} days, when k = {:.2f}, hpr = {:.5f}'.format(days,k,hpr))