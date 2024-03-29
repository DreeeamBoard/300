# 1초 주기로 비트코인 현재가 조회

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import pyupbit

form_class = uic.loadUiType("bitcoin.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        #self.pushButton.clicked.connect(self.inquiry)
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.inquiry)
        
    def inquiry(self) :
        cur_time = QTime.currentTime()
        str_time = cur_time.toString("hh:mm:ss")
        self.statusBar().showMessage(str_time)
        # price = pykorbit.get_current_price("BTC")
        price = pyupbit.get_current_price('KRW-BTC')
        self.lineEdit.setText(str(price))
        # print(price)
        
app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()