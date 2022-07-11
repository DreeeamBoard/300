import pyupbit
from fbprophet import Prophet
import matplotlib.pyplot as plt 
from fbprophet.plot import add_changepoints_to_plot
import pandas as pd

# https://colab.research.google.com/drive/1a4K3jCXQ1cN9QEumWYwcHqxBX3gEQGLO#scrollTo=bWPlaW3G5MmS

# 최근 200시간의 데이터
df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count = 200)

# 시간(ds)와 종가(y)값만 추출
df = df.reset_index()
df['ds'] = df['index']
df['y'] = df['close']
data = df[['ds','y']]

# 학습
model_1 = Prophet() # changepoint_prior_scale = 0.05이 default다
model_1.fit(data)

# 24시간 미래 예측
future_1 = model_1.make_future_dataframe(periods=24, freq='H')
forecast_1 = model_1.predict(future_1)

# 그래프 1
# 실제 가격 = 검은색 점
# 추세 = 선
fig_1_1 = model_1.plot(forecast_1)
#plt.show(fig_1_1)

# 그래프 2
fig_1_2 = model_1.plot_components(forecast_1)
#plt.show(fig_1_2)

fig_1_a = add_changepoints_to_plot(fig_1_1.gca(), model_1, forecast_1)
plt.show(fig_1_a)

# 만약 모델이 데이터의 trend를 잘 잡아내지 못하는 것 같다면
# changepoint_prior_scale 파라미터 값을 높여줘서 ->
# changepoint를 더 민감하게 감지하도록 할 수 있다
# changepoint란, trend가 변화하는 지점

model_2 = Prophet(changepoint_prior_scale = 0.9)
model_2.fit(data)
future_2 = model_2.make_future_dataframe(periods = 24, freq = 'H')
forecast_2 = model_2.predict(future_2)

fig_2_1 = model_2.plot(forecast_2)
#plt.show(fig_2_1)

fig_2_2 = model_2.plot_components(forecast_1)
#plt.show(fig_2_2)

fig_2_a = add_changepoints_to_plot(fig_2_1.gca(), model_2, forecast_2)
plt.show(fig_2_a)


#매수 시점의 가격
nowValue = pyupbit.get_current_price("KRW-BTC")
nowValue

forecast_ds = forecast['ds']
data_ds = data['ds']

test = pd.concat([forecast_ds, data_ds], axis=1)
test

#종가의 가격을 구함

#현재 시간이 자정 이전
closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
# forecast 데이터프레임은 x일 동안의 predict값도 포함되어 있지

#현재 시간이 자정 이후
if len(closeDf) == 0:
  closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]

#어쨋든 당일 종가
closeValue = closeDf['yhat'].values[0]
closeValue

#구체적인 가격
print("현재 시점 가격: ", nowValue)
print("종가의 가격: ", closeValue)

forecast
# yhat이 예측 가격이다