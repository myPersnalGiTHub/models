from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Deevia")
        MainWindow.resize(735, 646)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.deeviaImg = QtWidgets.QLabel(self.centralwidget)
        self.deeviaImg.setGeometry(QtCore.QRect(10,-10, 181, 101))
        self.deeviaImg.setText("")
        self.deeviaImg.setPixmap(QtGui.QPixmap("deevia.png"))
        self.deeviaImg.setScaledContents(True)
        self.deeviaImg.setObjectName("deeviaImg")
        
        '''self.graphWidget = QtWidgets.QLabel(pg.PlotWidget())
        self.graphWidget.setGeometry(QtCore.QRect(30,300, 181, 101))
        #self.setCentralWidget(self.graphWidget)
        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]
        self.graphWidget.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0))
        self.graphWidget.plot(hour, temperature, pen=pen)'''
        
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(10,120, 800, 520))
        self.picture.setText("")
        self.picture.setPixmap(QtGui.QPixmap("deevia.png"))
        self.picture.setScaledContents(True)
        self.picture.setObjectName("picture")

        self.picture_2 = QtWidgets.QLabel(self.centralwidget)
        self.picture_2.setGeometry(QtCore.QRect(850, 120, 700, 520))
        self.picture_2.setText("")
        self.picture_2.setPixmap(QtGui.QPixmap("deevia.png"))
        self.picture_2.setScaledContents(True)
        self.picture_2.setObjectName("picture_2")
        
        '''self.values = QtWidgets.QLabel(self.centralwidget)
        self.values.setGeometry(QtCore.QRect(580, 300, 111, 51))
        self.values.setObjectName("values")
        self.values.setText("Slab Values")'''

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 735, 21))
        self.menubar.setAutoFillBackground(False)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def showImg1(self,img):
        img = QtGui.QImage(img, int(1200),int(700), QtGui.QImage.Format_RGB888)
        self.picture.setPixmap(QtGui.QPixmap(img))
        return None
        
    def showImg2(self,img):
        img = QtGui.QImage(img, int(1200),int(700), QtGui.QImage.Format_RGB888)
        self.picture_2.setPixmap(QtGui.QPixmap(img))
        return None
        
    def showSlabDim(self,text):
        self.values.setText(str(test))

    def updateValue(self,val):
        self.values.setText(val)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Deevia", "Deevia"))
