# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'name_dialog.ui'
#
# Created: Thu Jul 31 17:40:37 2014
#      by: PyQt4 UI code generator 4.6.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_name(object):
    def setupUi(self, name):
        name.setObjectName("name")
        name.resize(336, 124)
        self.gridLayout = QtWidgets.QGridLayout(name)
        self.gridLayout.setObjectName("gridLayout")
        self.name_label = QtWidgets.QLabel(name)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_label.sizePolicy().hasHeightForWidth())
        self.name_label.setSizePolicy(sizePolicy)
        self.name_label.setObjectName("name_label")
        self.gridLayout.addWidget(self.name_label, 0, 0, 1, 1)
        self.nameBox = QtWidgets.QLineEdit(name)
        self.nameBox.setObjectName("nameBox")
        self.gridLayout.addWidget(self.nameBox, 1, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(name)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)

        self.retranslateUi(name)

        self.buttonBox.accepted.connect(name.accept)
        self.buttonBox.rejected.connect(name.reject)

        # QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), name.accept)
        # QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), name.reject)
        QtCore.QMetaObject.connectSlotsByName(name)

    def retranslateUi(self, name):
        name.setWindowTitle(
            QtWidgets.QApplication.translate("name", "Baseline scenario", None)
        )
        self.name_label.setText(
            QtWidgets.QApplication.translate(
                "name", "Enter a name for the new baseline scenario", None
            )
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    name = QtWidgets.QDialog()
    ui = Ui_name()
    ui.setupUi(name)
    name.show()
    sys.exit(app.exec_())
