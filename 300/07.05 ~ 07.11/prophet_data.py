# 2021년 데이터로 학습 -> 예측

import pyupbit
from fbprophet import Prophet

df = pyupbit.get_ohlcv('KRW-BTC', interval = 'day', count = 1000)
# 코랩에서는 1000일의 데이터가 호출되는데
# vscode에서는 200일 이상의 데이터는 호출이 안되네, 왜이러지
df['date'] = df.index
target_year = 2021
df_2021 = df[df['date'].dt.year == target_year]

df_2021 = df_2021.reset_index()
df_2021['ds'] = df_2021['index']
df_2021['y'] = df_2021['close']
data_2021 = df_2021[['ds','y']]

#학습
model = Prophet(changepoint_prior_scale = 0.05)
model.fit(data_2021)

# 200일 미래 예측
future = model.make_future_dataframe(periods = 200)
forecast = model.predict(future)
fig = model.plot(forecast)

####################################################################
# 2022년 이전의 모든 데이터로 학습 -> 예측
df = pyupbit.get_ohlcv('KRW-BTC', interval = 'day', to="20211231 09:00:00", count = 4000)

df = df.reset_index()
df['ds'] = df['index']
df['y'] = df['close']
data = df[['ds','y']]

#학습
model = Prophet(changepoint_prior_scale = 0.05)
model.fit(data)

# 300일 미래 예측
future = model.make_future_dataframe(periods = 300)
forecast = model.predict(future)
fig = model.plot(forecast)


#######################################################################
# 과거 모든 데이터로 학습 -> 300일 예측
# 20220711 기준 (to 참고)
df = pyupbit.get_ohlcv('KRW-BTC', interval = 'day', to="20220711 09:00:00", count = 4000)

df = df.reset_index()
df['ds'] = df['index']
df['y'] = df['close']
data = df[['ds','y']]

#학습
model = Prophet(changepoint_prior_scale = 0.05)
model.fit(data)

# 300일 미래 예측
future = model.make_future_dataframe(periods = 300)
forecast = model.predict(future)
fig = model.plot(forecast)