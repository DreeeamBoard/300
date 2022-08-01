# UI창에 가상화폐 종류, 현재가, 5일 이동평균, 상승/하락장 구현
# Prophet까지 구현해보자

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import pyupbit
from fbprophet import Prophet

tickers = ['KRW-BTC', 'KRW-ETH', 'KRW-BCH', 'KRW-ETC']
form_class = uic.loadUiType("bull.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        timer = QTimer(self)
        timer.start(1000)
        timer.timeout.connect(self.timeout)
            
    def get_market_infos(self, ticker):
        df = pyupbit.get_ohlcv(ticker)
        ma5 = df['close'].rolling(window=5).mean()
        last_ma5 = ma5[-2]
        price = pyupbit.get_current_price(ticker)

        state = None
        if price > last_ma5:
            state = "상승장"
        else:
            state = "하락장"
        return price, last_ma5, state
    
    def get_prophet(self, ticker):
        df = pyupbit.get_ohlcv(ticker)
        df2 = df.reset_index()
        df2['ds'] = df2['index']
        df2['y'] = df2['close']
        data = df2[['ds','y']]
        
        model = Prophet()
        model.fit(data)
        
        future = model.make_future_dataframe(periods=24, freq='H')
        forecast = model.predict(future)
        
        closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
        if len(closeDf) == 0:
            closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]
        
        closeValue = closeDf['yhat'].values[0]
        predicted_close_price = closeValue
        
        return predicted_close_price
        
    def timeout(self):
        for i, ticker in enumerate(tickers):
            ticker_item = QTableWidgetItem(ticker)
            self.tableWidget.setItem(i, 0, ticker_item)
            
            price, last_ma5, state = self.get_market_infos(ticker)
            predicted_close_price = self.get_prophet(ticker)
            
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(price)))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(last_ma5)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(state))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(predicted_close_price)))
            
    def inquiry(self) :
        cur_time = QTime.currentTime()
        str_time = cur_time.toString("hh:mm:ss")
        self.statusBar().showMessage(str_time)
        # price = pykorbit.get_current_price("BTC")
        price = pyupbit.get_current_price('KRW-BTC')
        self.lineEdit.setText(str(price))
            
app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()