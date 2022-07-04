import pyupbit
import numpy as np

# 이동 평균선 추가
# 모든 코인에 대해서 hpr값을 계산하고 best,worst coin 찾기
# k = 0.5로 고정

days = 10

def get_hpr(ticker) :
    try :
        df = pyupbit.get_ohlcv(ticker, count = days)
        df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
        # 5일 이동평균선
        df['range'] = (df['high'] - df['low']) * 0.5 # k = 0.5로 두자
        df['target'] = df['open'] + df['range'].shift(1)
        df['bull'] = df['open'] > df['ma5'] # 거래일의 시가가 전일 종가까지 계산된 5일 이동평균보다 높으면 bull컬럼에 true 저장

        fee = 0.0006

        df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
                            df['close'] / df['target'] - fee,
                            1)
        df['hpr'] = df['ror'].cumprod()
        return df['hpr'].iloc[-1]
    except :
        return 1

tickers = pyupbit.get_tickers(fiat='KRW')
hprs = []
hprss = []
for ticker in tickers :
    hpr = get_hpr(ticker)
    hprs.append((ticker, hpr))
    hprss.append(hpr)
    
sorted_hprs = sorted(hprs, key = lambda x:x[1], reverse = True)
rev_sorted_hprs = sorted(hprs, key = lambda x:x[1], reverse = False)
print('with MA')
print('for the last {} days'.format(days))
print('\n')
print('best coins : ', sorted_hprs[:5])
print('\n')
print('worst coins : ',rev_sorted_hprs[:5])
print('\n')
print('bitcoin : ', get_hpr('KRW-BTC'))
print('hpr mean = ', sum(hprss) / len(hprss))