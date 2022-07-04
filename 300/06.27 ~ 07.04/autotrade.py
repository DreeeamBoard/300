# 변동성 돌파 전략으로 자동매매
# 이동평균선 알고리즘도 추가

import time
import pyupbit
import datetime

f = open('api.txt')
lines = f.readlines()
access = lines[0].strip() # access key
secret = lines[1].strip() # secret key
f.close()

def get_target_price(ticker, k):
    # 매수 목표가 설정
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    # 2개만 조회. 어제와 오늘 데이터만 필요하므로
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker):
    #시작 시간 조회
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0] # 시간 값을 가져온다.
    return start_time

def get_ma15(ticker) :
    # 15일 이동 평균선 조회
    df = pyupbit.get_ohlcv(ticker, interval = 'day', count = 15)
    ma15 = df['close'].rolling(15).mean().iloc[-1]
    return ma15

def get_balance(ticker):
    # 잔고 조회
    # 굳이 이렇게 복잡하게 안하고 
    # balance = upbit.get_balance(ticker = 'KRW-BTC') 이렇게 해도 될듯?
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    # 현재가 조회
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start - Let's make money!")

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1) #하루를 더해서 다음날 9시가 마감
        target_price = get_target_price("KRW-BTC", k = 0.5)
        ma15 = get_ma15("KRW-BTC")
        current_price = get_current_price("KRW-BTC")
        
        print('now : ', now)
        print('target_price :', target_price)
        print('ma15 :', ma15)
        print('current_price :', current_price)
        
        if start_time < now < end_time - datetime.timedelta(seconds=30): # 9:00 < 현재 < 다음날 8시 59분 30초까지
            if current_price > target_price and current_price > ma15 :
                krw = get_balance("KRW")
                if krw > 5000: # 최소 거래 금액 5000원
                    print('lets buy!')
                    upbit.buy_market_order("KRW-BTC", krw*0.9995) # 수수료 0.05%
            else :
                print('be patient..')
                print('\n')
                    
        else: # 9시 되기 30초 전부터 계속 전량 매도 시도
            btc = get_balance("KRW-BTC")
            # if btc > 0.00008: # BTC이 5000원 이상이면
            print('lets sell!')
            upbit.sell_market_order("KRW-BTC", btc*0.9995)
        time.sleep(1)
        
    except Exception as e:
        print(e)
        time.sleep(1)