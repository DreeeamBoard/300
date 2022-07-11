import pyupbit
import numpy as np
from fbprophet import Prophet

days = 24 # for ~ days
df = pyupbit.get_ohlcv("KRW-BTC", count = days)
df['range'] = (df['high'] - df['low']) * 0.5 # k = 0.5
df['target'] = df['open'] + df['range'].shift(1)
df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
df['bull'] = df['open'] > df['ma5']
fee =  0.0006

df['ror'] = np.where((df['high'] > df['target'] > df['predicted']) & df['bull'],
                    df['close'] / df['target'] - fee,
                     1)

df['hpr'] = df['ror'].cumprod()
print("backtest for the last {} days".format(days))
print("HPR: ", df['hpr'].iloc[-1])
df.to_excel("prophet_backtest.xlsx")

df2 = df.reset_index()
df2['ds'] = df2['index'] # datestamp, YYYY-MM-DD HH:MM:SS
df2['y'] = df2['close'] # Numeric, 종가
data = df2[['ds','y']]

model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=24, freq='24H')
forecast = model.predict(future)

closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
if len(closeDf) == 0:
    closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]
closeValue = closeDf['yhat'].values[0]
predicted_close_price = closeValue
    
# 매일 target_price 계산
# 매 시간 간격으로 predicted_close_price를 계산하고.
# ⦁	Prophet 알고리즘 파악
# ⦁	Prophet으로 백테스트 진행 -> 실제 가격과 얼마나 유사하게 예측하는지 성능 체크해보기

# 추후 계획 :
# 1. Prophet을 이용했을 때 backtest 결과값 도출
# 2. Prophet 알고리즘 이해
# 3. 새로운 알고리즘, 트레이딩 기법 공부
# 4. 도버 강의