# Qt Designer로 만든 .ui파일을 사용해보자

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("thisone.ui")[0]

class MyWindow(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.btn_clicked)
        
    def btn_clicked(self):
        print("버튼 클릭")
        
app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()