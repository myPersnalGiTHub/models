from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
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
        self.deeviaImg.setPixmap(QtGui.QPixmap("deevia"))
        self.deeviaImg.setScaledContents(True)
        self.deeviaImg.setObjectName("deeviaImg")
                
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(10,80, 800, 450))
        self.picture.setText("")
        self.picture.setPixmap(QtGui.QPixmap("deevia"))
        self.picture.setScaledContents(True)
        self.picture.setObjectName("picture")

        self.picture_2 = QtWidgets.QLabel(self.centralwidget)
        self.picture_2.setGeometry(QtCore.QRect(850, 80, 750, 450))
        self.picture_2.setText("")
        self.picture_2.setPixmap(QtGui.QPixmap("deevia"))
        self.picture_2.setScaledContents(True)
        self.picture_2.setObjectName("picture_2")

        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(10, 525, 1570, 280))
        #self.graphWidget.setBackground('w')
        self.graphWidget.setLabel('left', 'Width (mm)', color='white', size=32)
        self.graphWidget.setLabel('bottom', 'frame number', color='white', size=38)
        self.graphWidget.setXRange(0, 150, padding=0)
        self.graphWidget.setYRange(950, 2100, padding=0)
        self.graphWidget.showGrid(x=True, y=True)
        
        
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
    
    def updateGraph(self,valCSV):

        #self.graphWidget.plot(clear = True)
    
        val = int(sum(valCSV)/len(valCSV))
        self.graphWidget.setYRange(val-30, val+30, padding=0)
        #self.graphWidget.plot(np.arange(len(valCSV)),valCSV)
        self.graphWidget.plot().setData(np.arange(len(valCSV)),valCSV)
        print('$$$plot graph$$$')
        return None

    def clearGraph(self):
        self.graphWidget.plot(clear = True)
        print("|___>graph cleared")
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Deevia", "Deevia"))
