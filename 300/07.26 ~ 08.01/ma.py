# 이동평균보다 현재가가 높다면 상승장
# 알고리즘 구현

import pyupbit

btc = pyupbit.get_ohlcv('KRW-BTC')
close = btc['close']

window = close.rolling(5)
ma5 = window.mean()

last_ma5 = ma5[-2]

btc_price = pyupbit.get_current_price('KRW-BTC')

if btc_price > last_ma5 :
    print("상승장")
else :
    print("하락장")