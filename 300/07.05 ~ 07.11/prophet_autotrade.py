import time
import pyupbit
import datetime
import schedule
from fbprophet import Prophet
from urllib.parse import urlparse

# 변동성 돌파 전략을 기반으로
# 돌파가격에 도달했을 때 (= 매수를 진행하는 시점에서) 앞으로의 가격을 예측한다
# -> 당일 종가의 예측 가격이 실제 매수가보다 높은지 판단한다
# -> 당일 종가의 예측 가격이 실제 매수가보다 높을 때만 매수를 진행한다

f = open('api.txt')
lines = f.readlines()
access = lines[0].strip()
secret = lines[1].strip()
f.close()

def get_target_price(ticker, k) :
    df = pyupbit.get_ohlcv(ticker, interval = 'day', count = 2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker) :
    df = pyupbit.get_ohlcv(ticker, interval = 'day', count = 1)
    start_time = df.index[0]
    return start_time

def get_ma15(ticker):
    df = pyupbit.get_ohlcv(ticker, interval = 'day', count = 15)
    ma15 = df['close'].rolling(15).mean().iloc[-1]
    return ma15

def get_balance(ticker) :
    balances = upbit.get_balances()
    for b in balances :
        if b['currency'] == ticker :
            if b['balance'] is not None :
                return float(b['balance'])
            else :
                return 0
    return 0

def get_current_price(ticker) :
    return pyupbit.get_current_price('KRW-BTC')
    # return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

predicted_close_price = 0
def predict_price(ticker):
    """Prophet으로 당일 종가 가격 예측"""
    global predicted_close_price
    
    df = pyupbit.get_ohlcv(ticker, interval="minute60")
    df = df.reset_index()
    df['ds'] = df['index'] # datestamp, YYYY-MM-DD HH:MM:SS
    df['y'] = df['close'] # Numeric, 종가
    data = df[['ds','y']]
    
    model = Prophet()
    model.fit(data)
    
    future = model.make_future_dataframe(periods=24, freq='H')
    # periods는 향후 몇 일 (or 주,월 등 단위)를 예측할 것인지
    # freq = 'H'는 hour
    forecast = model.predict(future)
    
    closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
    if len(closeDf) == 0:
        closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]
        
    closeValue = closeDf['yhat'].values[0]
    predicted_close_price = closeValue
predict_price("KRW-BTC")
#schedule.every().hour.do(lambda: predict_price("KRW-BTC")) # schedule로 한시간마다 함수가 돌아가도록.
schedule.every().minute.do(lambda: predict_price("KRW-BTC"))

# 로그인
upbit = pyupbit.Upbit(access, secret)
print('autotrade begins')

# 자동매매 시작
while True :
    try :
        now = datetime.datetime.now()
        start_time = get_start_time('KRW-BTC')
        end_time = start_time + datetime.timedelta(days=1)
        
        target_price = get_target_price('KRW-BTC', 0.5)
        current_price = get_current_price('KRW-BTC')
        ma15 = get_ma15("KRW-BTC")
        
        schedule.run_pending()
        
        if start_time < now < end_time - datetime.timedelta(seconds=10):
            if target_price < current_price and ma15 < current_price < predicted_close_price :
            # 목표가 < 현재가
            # 이동평균선 < 현재가 < 예측 종가
                krw = get_balance("KRW")
                if krw > 5000 :
                    upbit.buy_market_order('KRW-BTC', krw*0.9995)
            print('current : ', current_price, 'target : ', target_price, 'predicted close : ', predicted_close_price)
        else :
            btc = get_balance("KRW-BTC")
            if btc > 0.0000001 :
                upbit.sell_market_order('KRW-BTC', btc*0.9995)
        time.sleep(1)
    except Exception as e :
        print(e)
        time.sleep(1)