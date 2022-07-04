import pyupbit
import numpy as np

# 모든 코인에 대해서 hpr값을 계산하고 best,worst coin 찾기
# k = 0.5로 고정

days = 100

def get_hpr(ticker) :
    try :
        df = pyupbit.get_ohlcv(ticker, count = days)
        df['range'] = (df['high'] - df['low']) * 0.5 # k = 0.5로 두자
        df['target'] = df['open'] + df['range'].shift(1)

        fee = 0.0006

        df['ror'] = np.where(df['high'] > df['target'],
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
print('withOUT MA')
print('for the last {} days'.format(days))
print('\n')
print('best coins : ', sorted_hprs[:5])
print('\n')
print('worst coins : ',rev_sorted_hprs[:5])
print('\n')
# print('bitcoin : ', get_hpr('KRW-BTC'))
print('hpr mean = ', sum(hprss) / len(hprss))