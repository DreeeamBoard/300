import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

app = QApplication(sys.argv) # QApplication 객체 생성

# label = QLabel("Hello")
# label.show()

# btn = QPushButton('This is a button')
# btn.show()

# MyWindow 클래스 정의하기
class MyWindow(QMainWindow) :
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 200, 300, 400)
        # x축 위치, y축 위치, 윈도우의 너비, 윈도우의 높이
        self.setWindowTitle("here is the title")
        self.setWindowIcon(QIcon("icon.png"))
        
        btn1 = QPushButton("Button 1", self)
        btn1.move(10, 10) 
        # 버튼 위치 조절, x축, y축
        btn1.clicked.connect(self.btn_clicked)
        # 버튼1 객체가 클릭될 때, Mywindow 클래스에 정의된 btn1_clicked 메서드를 호출
        
        btn2 = QPushButton("Button 2", self)
        # 버튼 하나 더 추가
        btn2.move(10,40)
        
    def btn_clicked(self) :
        print("버튼 클릭")
        
window = MyWindow()
window.show()
app.exec_() # 이벤트 루프 생성
# 이벤트 루프는 루프를 돌고 있다가 사용자가 이벤트를 발생시키면 (ex 버튼 클릭) 이벤트에 연결된 메서드를 호출해주는 역할