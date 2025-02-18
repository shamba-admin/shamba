# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './shamba/disclaimer.ui'
#
# Created: Wed Jun 27 15:56:34 2012
#      by: PyQt4 UI code generator 4.6.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Disclaimer(object):
    def setupUi(self, Disclaimer):
        Disclaimer.setObjectName("Disclaimer")
        Disclaimer.resize(400, 524)
        self.verticalLayout = QtWidgets.QVBoxLayout(Disclaimer)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splashImage = QtWidgets.QLabel(Disclaimer)
        self.splashImage.setMinimumSize(QtCore.QSize(320, 240))
        self.splashImage.setAlignment(QtCore.Qt.AlignCenter)
        self.splashImage.setObjectName("splashImage")
        self.verticalLayout.addWidget(self.splashImage)
        self.scrollArea = QtWidgets.QScrollArea(Disclaimer)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget(self.scrollArea)
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 378, 223))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.disclaimerText = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.disclaimerText.setWordWrap(True)
        self.disclaimerText.setObjectName("disclaimerText")
        self.verticalLayout_2.addWidget(self.disclaimerText)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.buttonBox = QtWidgets.QDialogButtonBox(Disclaimer)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Disclaimer)
        self.buttonBox.accepted.connect(Disclaimer.accept)
        self.buttonBox.rejected.connect(Disclaimer.reject)

        # QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Disclaimer.accept)
        # QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Disclaimer.reject)
        QtCore.QMetaObject.connectSlotsByName(Disclaimer)

    def retranslateUi(self, Disclaimer):
        Disclaimer.setWindowTitle(
            QtWidgets.QApplication.translate("Disclaimer", "disclaimer", None)
        )
        self.disclaimerText.setText(
            QtWidgets.QApplication.translate("Disclaimer", "TextLabel", None)
        )
