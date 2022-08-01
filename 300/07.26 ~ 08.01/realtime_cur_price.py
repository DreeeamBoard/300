# 실시간으로 현재가 차트 그래프를 확인해보자

import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtChart import QLineSeries, QChart
from PyQt5.QtGui import QPainter
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QDateTimeAxis
from PyQt5.QtCore import Qt, QDateTime
import time
import pybithumb
from PyQt5.QtCore import QThread, pyqtSignal

class ChartWidget(QWidget):
     def __init__(self, parent=None, ticker="BTC"):
        super().__init__(parent)
        uic.loadUi("resource/chart.ui", self)
        self.ticker = ticker
         
        self.viewLimit = 128

        self.priceData = QLineSeries()
        self.priceData.append(0, 10)
        self.priceData.append(1, 20)
        self.priceData.append(2, 10)
 
        self.priceChart = QChart()
        self.priceChart.addSeries(self.priceData)
        self.priceView.setChart(self.priceChart)
        self.priceChart.legend().hide()
        
        axisX = QDateTimeAxis()
        axisX.setFormat("hh:mm:ss")
        axisX.setTickCount(4)
        dt = QDateTime.currentDateTime()
        axisX.setRange(dt, dt.addSecs(self.viewLimit))

        axisY = QValueAxis()
        axisY.setVisible(False)         
        self.priceChart.addAxis(axisX, Qt.AlignBottom)
        self.priceChart.addAxis(axisY, Qt.AlignRight)
        self.priceData.attachAxis(axisX)
        self.priceData.attachAxis(axisY)
        self.priceChart.layout().setContentsMargins(0, 0, 0, 0)
        self.priceView.setRenderHints(QPainter.Antialiasing)
 
class PriceWorker(QThread):
    dataSent = pyqtSignal(float)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
        self.alive = True

    def run(self):
         while self.alive:
            data  = pybithumb.get_current_price(self.ticker)
            time.sleep(1)
            self.dataSent.emit(data)
            
    def close(self):
         self.alive = False
         
    if __name__ == "__main__":
        import sys
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        cw = ChartWidget()
        cw.show()
        exit(app.exec_())