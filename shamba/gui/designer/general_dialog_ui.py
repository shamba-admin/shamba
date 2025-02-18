# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'general_dialog.ui'
#
# Created: Mon Aug  4 16:58:20 2014
#      by: PyQt4 UI code generator 4.6.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_general(object):
    def setupUi(self, general):
        general.setObjectName("general")
        general.resize(620, 580)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(general.sizePolicy().hasHeightForWidth())
        general.setSizePolicy(sizePolicy)
        general.setMinimumSize(QtCore.QSize(620, 580))
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(general)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.stackedWidget = QtWidgets.QStackedWidget(general)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.stackedWidget.sizePolicy().hasHeightForWidth()
        )
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.stackedWidget.setObjectName("stackedWidget")
        self.titlePage = QtWidgets.QWidget()
        self.titlePage.setObjectName("titlePage")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.titlePage)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.widget_4 = QtWidgets.QWidget(self.titlePage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setMinimumSize(QtCore.QSize(620, 175))
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_12 = QtWidgets.QLabel(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_4.addWidget(self.label_12)
        self.label_35 = QtWidgets.QLabel(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_35.sizePolicy().hasHeightForWidth())
        self.label_35.setSizePolicy(sizePolicy)
        self.label_35.setObjectName("label_35")
        self.verticalLayout_4.addWidget(self.label_35)
        self.verticalLayout_13.addWidget(self.widget_4)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.verticalLayout_13.addItem(spacerItem)
        self.stackedWidget.addWidget(self.titlePage)
        self.overviewPage = QtWidgets.QWidget()
        self.overviewPage.setObjectName("overviewPage")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.overviewPage)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget = QtWidgets.QWidget(self.overviewPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(620, 225))
        self.widget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMinimumSize(QtCore.QSize(0, 30))
        self.label_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_7.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.formLayout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout.setVerticalSpacing(15)
        self.formLayout.setObjectName("formLayout")
        spacerItem1 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.projectName = QtWidgets.QLineEdit(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.projectName.sizePolicy().hasHeightForWidth())
        self.projectName.setSizePolicy(sizePolicy)
        self.projectName.setMinimumSize(QtCore.QSize(323, 0))
        self.projectName.setMaximumSize(QtCore.QSize(323, 16777215))
        self.projectName.setMaxLength(1000)
        self.projectName.setObjectName("projectName")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.projectName)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.projectType = QtWidgets.QComboBox(self.widget)
        self.projectType.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.projectType.sizePolicy().hasHeightForWidth())
        self.projectType.setSizePolicy(sizePolicy)
        self.projectType.setMinimumSize(QtCore.QSize(175, 0))
        self.projectType.setMaximumSize(QtCore.QSize(175, 16777215))
        self.projectType.setEditable(False)
        self.projectType.setObjectName("projectType")
        self.projectType.addItem("")
        self.projectType.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.projectType)
        self.farmerNameLabel = QtWidgets.QLabel(self.widget)
        self.farmerNameLabel.setObjectName("farmerNameLabel")
        self.formLayout.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.farmerNameLabel
        )
        self.farmerName = QtWidgets.QLineEdit(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.farmerName.sizePolicy().hasHeightForWidth())
        self.farmerName.setSizePolicy(sizePolicy)
        self.farmerName.setMinimumSize(QtCore.QSize(323, 0))
        self.farmerName.setMaximumSize(QtCore.QSize(323, 16777215))
        self.farmerName.setObjectName("farmerName")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.farmerName)
        self.fieldNumLabel = QtWidgets.QLabel(self.widget)
        self.fieldNumLabel.setObjectName("fieldNumLabel")
        self.formLayout.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.fieldNumLabel
        )
        self.areaLabel = QtWidgets.QLabel(self.widget)
        self.areaLabel.setObjectName("areaLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.areaLabel)
        self.fieldNum = QtWidgets.QSpinBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fieldNum.sizePolicy().hasHeightForWidth())
        self.fieldNum.setSizePolicy(sizePolicy)
        self.fieldNum.setMinimumSize(QtCore.QSize(60, 0))
        self.fieldNum.setMaximum(99999)
        self.fieldNum.setObjectName("fieldNum")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.fieldNum)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(8)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.area = QtWidgets.QSpinBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.area.sizePolicy().hasHeightForWidth())
        self.area.setSizePolicy(sizePolicy)
        self.area.setMaximumSize(QtCore.QSize(60, 16777215))
        self.area.setMaximum(9999)
        self.area.setSingleStep(10)
        self.area.setObjectName("area")
        self.horizontalLayout_4.addWidget(self.area)
        self.areaText = QtWidgets.QLabel(self.widget)
        self.areaText.setObjectName("areaText")
        self.horizontalLayout_4.addWidget(self.areaText)
        self.formLayout.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_4
        )
        self.verticalLayout_2.addLayout(self.formLayout)
        self.verticalLayout_5.addWidget(self.widget)
        self.stackedWidget.addWidget(self.overviewPage)
        self.paramPage = QtWidgets.QWidget()
        self.paramPage.setObjectName("paramPage")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.paramPage)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_3 = QtWidgets.QWidget(self.paramPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setMinimumSize(QtCore.QSize(620, 300))
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_8 = QtWidgets.QLabel(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setMinimumSize(QtCore.QSize(0, 30))
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 0, 0, 1, 1)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_3.setVerticalSpacing(15)
        self.formLayout_3.setObjectName("formLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_3.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem2)
        self.label_13 = QtWidgets.QLabel(self.widget_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.verticalLayout_25 = QtWidgets.QVBoxLayout()
        self.verticalLayout_25.setSpacing(8)
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(8)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.latitude = QtWidgets.QDoubleSpinBox(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.latitude.sizePolicy().hasHeightForWidth())
        self.latitude.setSizePolicy(sizePolicy)
        self.latitude.setMinimumSize(QtCore.QSize(110, 0))
        self.latitude.setMaximumSize(QtCore.QSize(110, 16777215))
        self.latitude.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.latitude.setDecimals(6)
        self.latitude.setMinimum(-90.0)
        self.latitude.setMaximum(90.0)
        self.latitude.setProperty("value", -18.987214)
        self.latitude.setObjectName("latitude")
        self.horizontalLayout_2.addWidget(self.latitude)
        self.label_14 = QtWidgets.QLabel(self.widget_3)
        self.label_14.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_2.addWidget(self.label_14)
        self.verticalLayout_25.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(8)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.longitude = QtWidgets.QDoubleSpinBox(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.longitude.sizePolicy().hasHeightForWidth())
        self.longitude.setSizePolicy(sizePolicy)
        self.longitude.setMinimumSize(QtCore.QSize(110, 0))
        self.longitude.setMaximumSize(QtCore.QSize(110, 16777215))
        self.longitude.setDecimals(6)
        self.longitude.setMinimum(-180.0)
        self.longitude.setMaximum(180.0)
        self.longitude.setProperty("value", 34.174625)
        self.longitude.setObjectName("longitude")
        self.horizontalLayout_3.addWidget(self.longitude)
        self.label_15 = QtWidgets.QLabel(self.widget_3)
        self.label_15.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_3.addWidget(self.label_15)
        self.verticalLayout_25.addLayout(self.horizontalLayout_3)
        self.formLayout_3.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_25
        )
        self.label_18 = QtWidgets.QLabel(self.widget_3)
        self.label_18.setObjectName("label_18")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(8)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.mitigationYears = QtWidgets.QSpinBox(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.mitigationYears.sizePolicy().hasHeightForWidth()
        )
        self.mitigationYears.setSizePolicy(sizePolicy)
        self.mitigationYears.setMinimumSize(QtCore.QSize(60, 0))
        self.mitigationYears.setMaximumSize(QtCore.QSize(60, 16777215))
        self.mitigationYears.setMaximum(100)
        self.mitigationYears.setSingleStep(10)
        self.mitigationYears.setProperty("value", 30)
        self.mitigationYears.setObjectName("mitigationYears")
        self.horizontalLayout_7.addWidget(self.mitigationYears)
        self.label_22 = QtWidgets.QLabel(self.widget_3)
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_7.addWidget(self.label_22)
        self.formLayout_3.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_7
        )
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.gridLayout_3.addLayout(self.formLayout_3, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.widget_3)
        self.stackedWidget.addWidget(self.paramPage)
        self.climatePage = QtWidgets.QWidget()
        self.climatePage.setObjectName("climatePage")
        self.verticalLayout_31 = QtWidgets.QVBoxLayout(self.climatePage)
        self.verticalLayout_31.setObjectName("verticalLayout_31")
        self.climateWidget = QtWidgets.QWidget(self.climatePage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climateWidget.sizePolicy().hasHeightForWidth()
        )
        self.climateWidget.setSizePolicy(sizePolicy)
        self.climateWidget.setMinimumSize(QtCore.QSize(620, 150))
        self.climateWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.climateWidget.setObjectName("climateWidget")
        self.gridLayout_19 = QtWidgets.QGridLayout(self.climateWidget)
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.label_16 = QtWidgets.QLabel(self.climateWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setMinimumSize(QtCore.QSize(0, 30))
        self.label_16.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_16.setObjectName("label_16")
        self.gridLayout_19.addWidget(self.label_16, 0, 0, 1, 1)
        self.formLayout_11 = QtWidgets.QFormLayout()
        self.formLayout_11.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_11.setVerticalSpacing(15)
        self.formLayout_11.setObjectName("formLayout_11")
        spacerItem3 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_11.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem3)
        self.verticalLayout_22 = QtWidgets.QVBoxLayout()
        self.verticalLayout_22.setSpacing(8)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.climFromCRU = QtWidgets.QRadioButton(self.climateWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.climFromCRU.sizePolicy().hasHeightForWidth())
        self.climFromCRU.setSizePolicy(sizePolicy)
        self.climFromCRU.setObjectName("climFromCRU")
        self.verticalLayout_22.addWidget(self.climFromCRU)
        self.climFromCsv = QtWidgets.QRadioButton(self.climateWidget)
        self.climFromCsv.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.climFromCsv.sizePolicy().hasHeightForWidth())
        self.climFromCsv.setSizePolicy(sizePolicy)
        self.climFromCsv.setObjectName("climFromCsv")
        self.verticalLayout_22.addWidget(self.climFromCsv)
        self.climFromCustom = QtWidgets.QRadioButton(self.climateWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climFromCustom.sizePolicy().hasHeightForWidth()
        )
        self.climFromCustom.setSizePolicy(sizePolicy)
        self.climFromCustom.setObjectName("climFromCustom")
        self.verticalLayout_22.addWidget(self.climFromCustom)
        self.formLayout_11.setLayout(
            1, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_22
        )
        self.climCsvLabel = QtWidgets.QLabel(self.climateWidget)
        self.climCsvLabel.setObjectName("climCsvLabel")
        self.formLayout_11.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.climCsvLabel
        )
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(8)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.climCsvBrowseBox = QtWidgets.QLineEdit(self.climateWidget)
        self.climCsvBrowseBox.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climCsvBrowseBox.sizePolicy().hasHeightForWidth()
        )
        self.climCsvBrowseBox.setSizePolicy(sizePolicy)
        self.climCsvBrowseBox.setMinimumSize(QtCore.QSize(230, 0))
        self.climCsvBrowseBox.setMaximumSize(QtCore.QSize(230, 16777215))
        self.climCsvBrowseBox.setMaxLength(1000)
        self.climCsvBrowseBox.setObjectName("climCsvBrowseBox")
        self.horizontalLayout.addWidget(self.climCsvBrowseBox)
        self.climCsvBrowseButton = QtWidgets.QPushButton(self.climateWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climCsvBrowseButton.sizePolicy().hasHeightForWidth()
        )
        self.climCsvBrowseButton.setSizePolicy(sizePolicy)
        self.climCsvBrowseButton.setMinimumSize(QtCore.QSize(85, 0))
        self.climCsvBrowseButton.setMaximumSize(QtCore.QSize(85, 16777215))
        self.climCsvBrowseButton.setObjectName("climCsvBrowseButton")
        self.horizontalLayout.addWidget(self.climCsvBrowseButton)
        self.formLayout_11.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout
        )
        self.gridLayout_19.addLayout(self.formLayout_11, 1, 0, 1, 1)
        self.verticalLayout_31.addWidget(self.climateWidget)
        self.stackedWidget.addWidget(self.climatePage)
        self.climateTablePage = QtWidgets.QWidget()
        self.climateTablePage.setObjectName("climateTablePage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.climateTablePage)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.climateTableWidget_2 = QtWidgets.QWidget(self.climateTablePage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climateTableWidget_2.sizePolicy().hasHeightForWidth()
        )
        self.climateTableWidget_2.setSizePolicy(sizePolicy)
        self.climateTableWidget_2.setMinimumSize(QtCore.QSize(620, 70))
        self.climateTableWidget_2.setMaximumSize(QtCore.QSize(16777215, 100))
        self.climateTableWidget_2.setObjectName("climateTableWidget_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.climateTableWidget_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_23 = QtWidgets.QLabel(self.climateTableWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setMinimumSize(QtCore.QSize(0, 30))
        self.label_23.setObjectName("label_23")
        self.gridLayout_4.addWidget(self.label_23, 0, 0, 1, 1)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_24 = QtWidgets.QLabel(self.climateTableWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setMinimumSize(QtCore.QSize(240, 0))
        self.label_24.setMaximumSize(QtCore.QSize(240, 16777215))
        self.label_24.setObjectName("label_24")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setSpacing(8)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.openPanEvapCheck = QtWidgets.QRadioButton(self.climateTableWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.openPanEvapCheck.sizePolicy().hasHeightForWidth()
        )
        self.openPanEvapCheck.setSizePolicy(sizePolicy)
        self.openPanEvapCheck.setObjectName("openPanEvapCheck")
        self.horizontalLayout_10.addWidget(self.openPanEvapCheck)
        self.evapotransCheck = QtWidgets.QRadioButton(self.climateTableWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.evapotransCheck.sizePolicy().hasHeightForWidth()
        )
        self.evapotransCheck.setSizePolicy(sizePolicy)
        self.evapotransCheck.setObjectName("evapotransCheck")
        self.horizontalLayout_10.addWidget(self.evapotransCheck)
        self.formLayout_4.setLayout(
            0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_10
        )
        self.gridLayout_4.addLayout(self.formLayout_4, 1, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.climateTableWidget_2)
        self.climateTableWidget = QtWidgets.QWidget(self.climateTablePage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.climateTableWidget.sizePolicy().hasHeightForWidth()
        )
        self.climateTableWidget.setSizePolicy(sizePolicy)
        self.climateTableWidget.setMinimumSize(QtCore.QSize(400, 400))
        self.climateTableWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.climateTableWidget.setObjectName("climateTableWidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.climateTableWidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.climateTable = QtWidgets.QTableWidget(self.climateTableWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.climateTable.sizePolicy().hasHeightForWidth())
        self.climateTable.setSizePolicy(sizePolicy)
        self.climateTable.setMinimumSize(QtCore.QSize(405, 405))
        self.climateTable.setMaximumSize(QtCore.QSize(691, 16777215))
        self.climateTable.setObjectName("climateTable")
        self.climateTable.setColumnCount(3)
        self.climateTable.setRowCount(12)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.climateTable.setHorizontalHeaderItem(2, item)
        self.climateTable.horizontalHeader().setDefaultSectionSize(200)
        self.climateTable.horizontalHeader().setMinimumSectionSize(200)
        self.climateTable.verticalHeader().setDefaultSectionSize(30)
        self.climateTable.verticalHeader().setMinimumSectionSize(21)
        self.gridLayout_5.addWidget(self.climateTable, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.climateTableWidget)
        self.stackedWidget.addWidget(self.climateTablePage)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.widget_5 = QtWidgets.QWidget(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy)
        self.widget_5.setMinimumSize(QtCore.QSize(620, 175))
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_17 = QtWidgets.QLabel(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_6.addWidget(self.label_17)
        self.label_36 = QtWidgets.QLabel(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_36.sizePolicy().hasHeightForWidth())
        self.label_36.setSizePolicy(sizePolicy)
        self.label_36.setObjectName("label_36")
        self.verticalLayout_6.addWidget(self.label_36)
        self.verticalLayout_8.addWidget(self.widget_5)
        spacerItem4 = QtWidgets.QSpacerItem(
            20,
            247,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.verticalLayout_8.addItem(spacerItem4)
        self.stackedWidget.addWidget(self.page)
        self.verticalLayout_7.addWidget(self.stackedWidget)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.helpButton = QtWidgets.QDialogButtonBox(general)
        self.helpButton.setStandardButtons(QtWidgets.QDialogButtonBox.Help)
        self.helpButton.setObjectName("helpButton")
        self.horizontalLayout_9.addWidget(self.helpButton)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.backButton = QtWidgets.QPushButton(general)
        self.backButton.setCheckable(False)
        self.backButton.setObjectName("backButton")
        self.horizontalLayout_8.addWidget(self.backButton)
        self.nextButton = QtWidgets.QPushButton(general)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout_8.addWidget(self.nextButton)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_8)
        self.verticalLayout_7.addLayout(self.horizontalLayout_9)

        self.retranslateUi(general)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(general)

    def retranslateUi(self, general):
        general.setWindowTitle(
            QtWidgets.QApplication.translate("general", "General Information", None)
        )
        self.label_12.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:18pt; font-weight:600;">GENERAL INFORMATION</span></p></body></html>',
                None,
            )
        )
        self.label_35.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p align="justify" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">The following screens will prompt you to enter general and</span></p>\n'
                '<p align="justify" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">geographical information about the new project.</span></p></body></html>',
                None,
            )
        )
        self.label_7.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Project Overview</span></p></body></html>',
                None,
            )
        )
        self.label.setText(
            QtWidgets.QApplication.translate("general", "Project Name", None)
        )
        self.label_4.setText(
            QtWidgets.QApplication.translate("general", "Level of Assessment", None)
        )
        self.projectType.setItemText(
            0, QtWidgets.QApplication.translate("general", "Project", None)
        )
        self.projectType.setItemText(
            1, QtWidgets.QApplication.translate("general", "Individual field", None)
        )
        self.farmerNameLabel.setText(
            QtWidgets.QApplication.translate("general", "Farmer Name", None)
        )
        self.fieldNumLabel.setText(
            QtWidgets.QApplication.translate("general", "Field Number", None)
        )
        self.areaLabel.setText(
            QtWidgets.QApplication.translate("general", "Field Area", None)
        )
        self.areaText.setText(QtWidgets.QApplication.translate("general", "ha", None))
        self.label_8.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Location and Project Periods</span></p></body></html>',
                None,
            )
        )
        self.label_13.setText(
            QtWidgets.QApplication.translate("general", "Location of Project", None)
        )
        self.label_14.setText(
            QtWidgets.QApplication.translate("general", "latitude", None)
        )
        self.label_15.setText(
            QtWidgets.QApplication.translate("general", "longitude", None)
        )
        self.label_18.setText(
            QtWidgets.QApplication.translate("general", "Quantification Period", None)
        )
        self.label_22.setText(
            QtWidgets.QApplication.translate("general", "years", None)
        )
        self.label_16.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Climate Data</span></p></body></html>',
                None,
            )
        )
        self.climFromCRU.setText(
            QtWidgets.QApplication.translate(
                "general",
                "Use SHAMBA default data for this\n" "location (recommended)",
                None,
            )
        )
        self.climFromCsv.setText(
            QtWidgets.QApplication.translate("general", "Load climate data", None)
        )
        self.climFromCustom.setText(
            QtWidgets.QApplication.translate(
                "general", "Enter climate data directly", None
            )
        )
        self.climCsvLabel.setText(
            QtWidgets.QApplication.translate("general", "Select a file to load:", None)
        )
        self.climCsvBrowseButton.setText(
            QtWidgets.QApplication.translate("general", "Browse", None)
        )
        self.label_23.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Custom Climate Data</span></p></body></html>',
                None,
            )
        )
        self.label_24.setText(
            QtWidgets.QApplication.translate("general", "Which are you entering?", None)
        )
        self.openPanEvapCheck.setText(
            QtWidgets.QApplication.translate("general", "Open-pan Evaporation", None)
        )
        self.evapotransCheck.setText(
            QtWidgets.QApplication.translate("general", "Evapotranspiration", None)
        )
        self.climateTable.verticalHeaderItem(0).setText(
            QtWidgets.QApplication.translate("general", "January", None)
        )
        self.climateTable.verticalHeaderItem(1).setText(
            QtWidgets.QApplication.translate("general", "February", None)
        )
        self.climateTable.verticalHeaderItem(2).setText(
            QtWidgets.QApplication.translate("general", "March", None)
        )
        self.climateTable.verticalHeaderItem(3).setText(
            QtWidgets.QApplication.translate("general", "April", None)
        )
        self.climateTable.verticalHeaderItem(4).setText(
            QtWidgets.QApplication.translate("general", "May", None)
        )
        self.climateTable.verticalHeaderItem(5).setText(
            QtWidgets.QApplication.translate("general", "June", None)
        )
        self.climateTable.verticalHeaderItem(6).setText(
            QtWidgets.QApplication.translate("general", "July", None)
        )
        self.climateTable.verticalHeaderItem(7).setText(
            QtWidgets.QApplication.translate("general", "August", None)
        )
        self.climateTable.verticalHeaderItem(8).setText(
            QtWidgets.QApplication.translate("general", "September", None)
        )
        self.climateTable.verticalHeaderItem(9).setText(
            QtWidgets.QApplication.translate("general", "October", None)
        )
        self.climateTable.verticalHeaderItem(10).setText(
            QtWidgets.QApplication.translate("general", "November", None)
        )
        self.climateTable.verticalHeaderItem(11).setText(
            QtWidgets.QApplication.translate("general", "December", None)
        )
        self.climateTable.horizontalHeaderItem(0).setText(
            QtWidgets.QApplication.translate("general", "Temperature (C)", None)
        )
        self.climateTable.horizontalHeaderItem(1).setText(
            QtWidgets.QApplication.translate("general", "Precipitation (mm)", None)
        )
        self.climateTable.horizontalHeaderItem(2).setText(
            QtWidgets.QApplication.translate(
                "general", "Open-pan Evaporation (mm)", None
            )
        )
        self.label_17.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:18pt; font-weight:600;">END OF GENERAL INFORMATION</span></p></body></html>',
                None,
            )
        )
        self.label_36.setText(
            QtWidgets.QApplication.translate(
                "general",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">Click the &quot;Save&quot; button to save information</span></p>\n'
                '<p align="justify" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">or click &quot;Back&quot; to review.</span></p></body></html>',
                None,
            )
        )
        self.backButton.setText(
            QtWidgets.QApplication.translate("general", "Back", None)
        )
        self.nextButton.setText(
            QtWidgets.QApplication.translate("general", "Next", None)
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    general = QtWidgets.QDialog()
    ui = Ui_general()
    ui.setupUi(general)
    general.show()
    sys.exit(app.exec_())
