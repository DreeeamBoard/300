import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
# import pykorbit
# 책에는 코빗을 사용했는데, 업비트를 써도 무방한듯?
import pyupbit

form_class = uic.loadUiType("bitcoin.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.inquiry)
    
    def inquiry(self) :
        # price = pykorbit.get_current_price("BTC")
        price = pyupbit.get_current_price('KRW-BTC')
        self.lineEdit.setText(str(price))
        # print(price)
        
app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()