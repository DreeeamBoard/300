import pyupbit
import numpy as np

# open, high, low, close, volume
days = 90 # for ~ days
df = pyupbit.get_ohlcv("KRW-BTC", count = days)

# 변동폭 구하기
df['range'] = (df['high'] - df['low']) * 0.5 # k = 0.5

# 다음날이기 때문에 shift(1)
# 다음날의 시가에 range를 구해줘서 목표가를 구한다
df['target'] = df['open'] + df['range'].shift(1)

df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
df['bull'] = df['open'] > df['ma5']
fee =  0.0006

# ror (rate of returns, 수익률), np.where(조건문, 참일때 값, 거짓일때 값)
# 여기서 왜 high > target을 했지? price > target 아닌가?
df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
                    df['close'] / df['target'] - fee,
                     1)

# df['ror'] = np.where((df['high'] > df['target']),
#                      df['close'] / df['target'] - fee,
#                       1)
# 누적 곱 계산 (cumprod) -> 누적 수익률
# holding period return (기간수익률)
df['hpr'] = df['ror'].cumprod()

# draw down 계산 (누적 최대 값과 현재 hpr 차이 / 누적 최대값 * 100)
# df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100

print("backtest for the last {} days".format(days))
# max draw down
# print("MDD(%): ", df['dd'].max())
print("HPR: ", df['hpr'].iloc[-1])
df.to_excel("backtest.xlsx")

# 하락장 기준
# 그냥 홀딩 하는 것 보다 변동성 돌파 전략을 썼을 때, 손실이 적었다.

# 하락장에서도 수익을 낼 수 있는 알고리즘..?