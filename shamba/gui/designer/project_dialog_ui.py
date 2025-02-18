# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'project_dialog.ui'
#
# Created: Mon Aug  4 16:58:14 2014
#      by: PyQt4 UI code generator 4.6.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_project(object):
    def setupUi(self, project):
        project.setObjectName("project")
        project.resize(620, 580)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(project.sizePolicy().hasHeightForWidth())
        project.setSizePolicy(sizePolicy)
        project.setMinimumSize(QtCore.QSize(620, 580))
        self.verticalLayout = QtWidgets.QVBoxLayout(project)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.stackedWidget = QtWidgets.QStackedWidget(project)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.stackedWidget.sizePolicy().hasHeightForWidth()
        )
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QtCore.QSize(602, 0))
        self.stackedWidget.setObjectName("stackedWidget")
        self.titlePage = QtWidgets.QWidget()
        self.titlePage.setObjectName("titlePage")
        self.verticalLayout_32 = QtWidgets.QVBoxLayout(self.titlePage)
        self.verticalLayout_32.setObjectName("verticalLayout_32")
        self.widget_5 = QtWidgets.QWidget(self.titlePage)
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
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.title_label = QtWidgets.QLabel(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title_label.sizePolicy().hasHeightForWidth())
        self.title_label.setSizePolicy(sizePolicy)
        self.title_label.setObjectName("title_label")
        self.verticalLayout_7.addWidget(self.title_label)
        self.subtitle_label = QtWidgets.QLabel(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.subtitle_label.sizePolicy().hasHeightForWidth()
        )
        self.subtitle_label.setSizePolicy(sizePolicy)
        self.subtitle_label.setObjectName("subtitle_label")
        self.verticalLayout_7.addWidget(self.subtitle_label)
        self.verticalLayout_32.addWidget(self.widget_5)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_32.addItem(spacerItem)
        self.stackedWidget.addWidget(self.titlePage)
        self.infoPage = QtWidgets.QWidget()
        self.infoPage.setObjectName("infoPage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.infoPage)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.baselineOverviewWidget = QtWidgets.QWidget(self.infoPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.baselineOverviewWidget.sizePolicy().hasHeightForWidth()
        )
        self.baselineOverviewWidget.setSizePolicy(sizePolicy)
        self.baselineOverviewWidget.setMinimumSize(QtCore.QSize(620, 150))
        self.baselineOverviewWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.baselineOverviewWidget.setObjectName("baselineOverviewWidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.baselineOverviewWidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_11 = QtWidgets.QLabel(self.baselineOverviewWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(0, 30))
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_11.setObjectName("label_11")
        self.verticalLayout_5.addWidget(self.label_11)
        self.formLayout_6 = QtWidgets.QFormLayout()
        self.formLayout_6.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_6.setContentsMargins(-1, -1, -1, 9)
        self.formLayout_6.setHorizontalSpacing(6)
        self.formLayout_6.setVerticalSpacing(15)
        self.formLayout_6.setObjectName("formLayout_6")
        spacerItem1 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_6.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem1)
        self.cropNumLabel = QtWidgets.QLabel(self.baselineOverviewWidget)
        self.cropNumLabel.setEnabled(True)
        self.cropNumLabel.setObjectName("cropNumLabel")
        self.formLayout_6.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.cropNumLabel
        )
        self.cropNum = QtWidgets.QSpinBox(self.baselineOverviewWidget)
        self.cropNum.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropNum.sizePolicy().hasHeightForWidth())
        self.cropNum.setSizePolicy(sizePolicy)
        self.cropNum.setMinimumSize(QtCore.QSize(50, 0))
        self.cropNum.setMinimum(0)
        self.cropNum.setMaximum(10)
        self.cropNum.setProperty("value", 1)
        self.cropNum.setObjectName("cropNum")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cropNum)
        self.treeNumLabel = QtWidgets.QLabel(self.baselineOverviewWidget)
        self.treeNumLabel.setEnabled(True)
        self.treeNumLabel.setObjectName("treeNumLabel")
        self.formLayout_6.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.treeNumLabel
        )
        self.treeNum = QtWidgets.QSpinBox(self.baselineOverviewWidget)
        self.treeNum.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeNum.sizePolicy().hasHeightForWidth())
        self.treeNum.setSizePolicy(sizePolicy)
        self.treeNum.setMinimumSize(QtCore.QSize(50, 0))
        self.treeNum.setMinimum(0)
        self.treeNum.setMaximum(10)
        self.treeNum.setProperty("value", 0)
        self.treeNum.setObjectName("treeNum")
        self.formLayout_6.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.treeNum)
        self.litterNumLabel = QtWidgets.QLabel(self.baselineOverviewWidget)
        self.litterNumLabel.setEnabled(True)
        self.litterNumLabel.setObjectName("litterNumLabel")
        self.formLayout_6.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.litterNumLabel
        )
        self.litterNum = QtWidgets.QSpinBox(self.baselineOverviewWidget)
        self.litterNum.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.litterNum.sizePolicy().hasHeightForWidth())
        self.litterNum.setSizePolicy(sizePolicy)
        self.litterNum.setMinimumSize(QtCore.QSize(50, 0))
        self.litterNum.setMaximum(2)
        self.litterNum.setProperty("value", 0)
        self.litterNum.setObjectName("litterNum")
        self.formLayout_6.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.litterNum)
        self.label = QtWidgets.QLabel(self.baselineOverviewWidget)
        self.label.setObjectName("label")
        self.formLayout_6.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label)
        self.fertNum = QtWidgets.QSpinBox(self.baselineOverviewWidget)
        self.fertNum.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertNum.sizePolicy().hasHeightForWidth())
        self.fertNum.setSizePolicy(sizePolicy)
        self.fertNum.setMinimumSize(QtCore.QSize(50, 0))
        self.fertNum.setMinimum(0)
        self.fertNum.setMaximum(2)
        self.fertNum.setProperty("value", 0)
        self.fertNum.setObjectName("fertNum")
        self.formLayout_6.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.fertNum)
        self.verticalLayout_5.addLayout(self.formLayout_6)
        self.verticalLayout_3.addWidget(self.baselineOverviewWidget)
        self.stackedWidget.addWidget(self.infoPage)
        self.soilPage = QtWidgets.QWidget()
        self.soilPage.setObjectName("soilPage")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.soilPage)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.soilWidget = QtWidgets.QWidget(self.soilPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.soilWidget.sizePolicy().hasHeightForWidth())
        self.soilWidget.setSizePolicy(sizePolicy)
        self.soilWidget.setMinimumSize(QtCore.QSize(620, 150))
        self.soilWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.soilWidget.setObjectName("soilWidget")
        self.gridLayout_20 = QtWidgets.QGridLayout(self.soilWidget)
        self.gridLayout_20.setObjectName("gridLayout_20")
        self.label_10 = QtWidgets.QLabel(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setMinimumSize(QtCore.QSize(0, 30))
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_10.setObjectName("label_10")
        self.gridLayout_20.addWidget(self.label_10, 0, 0, 1, 1)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_5.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout_5.setLabelAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.formLayout_5.setFormAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.formLayout_5.setVerticalSpacing(15)
        self.formLayout_5.setObjectName("formLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_5.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem2)
        self.verticalLayout_23 = QtWidgets.QVBoxLayout()
        self.verticalLayout_23.setSpacing(8)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.soilFromHWSD = QtWidgets.QRadioButton(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.soilFromHWSD.sizePolicy().hasHeightForWidth())
        self.soilFromHWSD.setSizePolicy(sizePolicy)
        self.soilFromHWSD.setObjectName("soilFromHWSD")
        self.verticalLayout_23.addWidget(self.soilFromHWSD)
        self.soilFromCsv = QtWidgets.QRadioButton(self.soilWidget)
        self.soilFromCsv.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.soilFromCsv.sizePolicy().hasHeightForWidth())
        self.soilFromCsv.setSizePolicy(sizePolicy)
        self.soilFromCsv.setObjectName("soilFromCsv")
        self.verticalLayout_23.addWidget(self.soilFromCsv)
        self.soilFromCustom = QtWidgets.QRadioButton(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.soilFromCustom.sizePolicy().hasHeightForWidth()
        )
        self.soilFromCustom.setSizePolicy(sizePolicy)
        self.soilFromCustom.setObjectName("soilFromCustom")
        self.verticalLayout_23.addWidget(self.soilFromCustom)
        self.formLayout_5.setLayout(
            1, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_23
        )
        self.soilCsvLabel = QtWidgets.QLabel(self.soilWidget)
        self.soilCsvLabel.setObjectName("soilCsvLabel")
        self.formLayout_5.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.soilCsvLabel
        )
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.soilCsvBrowseBox = QtWidgets.QLineEdit(self.soilWidget)
        self.soilCsvBrowseBox.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.soilCsvBrowseBox.sizePolicy().hasHeightForWidth()
        )
        self.soilCsvBrowseBox.setSizePolicy(sizePolicy)
        self.soilCsvBrowseBox.setMinimumSize(QtCore.QSize(230, 0))
        self.soilCsvBrowseBox.setMaximumSize(QtCore.QSize(230, 16777215))
        self.soilCsvBrowseBox.setMaxLength(1000)
        self.soilCsvBrowseBox.setObjectName("soilCsvBrowseBox")
        self.horizontalLayout_4.addWidget(self.soilCsvBrowseBox)
        self.soilCsvBrowseButton = QtWidgets.QPushButton(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.soilCsvBrowseButton.sizePolicy().hasHeightForWidth()
        )
        self.soilCsvBrowseButton.setSizePolicy(sizePolicy)
        self.soilCsvBrowseButton.setMinimumSize(QtCore.QSize(85, 0))
        self.soilCsvBrowseButton.setMaximumSize(QtCore.QSize(85, 16777215))
        self.soilCsvBrowseButton.setObjectName("soilCsvBrowseButton")
        self.horizontalLayout_4.addWidget(self.soilCsvBrowseButton)
        self.formLayout_5.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_4
        )
        self.soilCy0Label = QtWidgets.QLabel(self.soilWidget)
        self.soilCy0Label.setObjectName("soilCy0Label")
        self.formLayout_5.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.soilCy0Label
        )
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setSpacing(8)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.Cy0Input = QtWidgets.QDoubleSpinBox(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Cy0Input.sizePolicy().hasHeightForWidth())
        self.Cy0Input.setSizePolicy(sizePolicy)
        self.Cy0Input.setMinimumSize(QtCore.QSize(80, 0))
        self.Cy0Input.setMaximumSize(QtCore.QSize(80, 16777215))
        self.Cy0Input.setDecimals(1)
        self.Cy0Input.setMaximum(200.0)
        self.Cy0Input.setSingleStep(0.5)
        self.Cy0Input.setObjectName("Cy0Input")
        self.horizontalLayout_11.addWidget(self.Cy0Input)
        self.soilCy0Unit = QtWidgets.QLabel(self.soilWidget)
        self.soilCy0Unit.setObjectName("soilCy0Unit")
        self.horizontalLayout_11.addWidget(self.soilCy0Unit)
        self.formLayout_5.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_11
        )
        self.soilClayLabel = QtWidgets.QLabel(self.soilWidget)
        self.soilClayLabel.setObjectName("soilClayLabel")
        self.formLayout_5.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.soilClayLabel
        )
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setSpacing(8)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.clayInput = QtWidgets.QDoubleSpinBox(self.soilWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clayInput.sizePolicy().hasHeightForWidth())
        self.clayInput.setSizePolicy(sizePolicy)
        self.clayInput.setMinimumSize(QtCore.QSize(80, 0))
        self.clayInput.setMaximumSize(QtCore.QSize(80, 16777215))
        self.clayInput.setDecimals(1)
        self.clayInput.setMaximum(100.0)
        self.clayInput.setSingleStep(10.0)
        self.clayInput.setObjectName("clayInput")
        self.horizontalLayout_12.addWidget(self.clayInput)
        self.soilClayUnit = QtWidgets.QLabel(self.soilWidget)
        self.soilClayUnit.setObjectName("soilClayUnit")
        self.horizontalLayout_12.addWidget(self.soilClayUnit)
        self.formLayout_5.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_12
        )
        self.gridLayout_20.addLayout(self.formLayout_5, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.soilWidget)
        self.stackedWidget.addWidget(self.soilPage)
        self.soilMgmtPage = QtWidgets.QWidget()
        self.soilMgmtPage.setObjectName("soilMgmtPage")
        self.verticalLayout_34 = QtWidgets.QVBoxLayout(self.soilMgmtPage)
        self.verticalLayout_34.setObjectName("verticalLayout_34")
        self.soilMgmtWidget_base = QtWidgets.QWidget(self.soilMgmtPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.soilMgmtWidget_base.sizePolicy().hasHeightForWidth()
        )
        self.soilMgmtWidget_base.setSizePolicy(sizePolicy)
        self.soilMgmtWidget_base.setMinimumSize(QtCore.QSize(620, 200))
        self.soilMgmtWidget_base.setObjectName("soilMgmtWidget_base")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.soilMgmtWidget_base)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_36 = QtWidgets.QLabel(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_36.sizePolicy().hasHeightForWidth())
        self.label_36.setSizePolicy(sizePolicy)
        self.label_36.setMinimumSize(QtCore.QSize(0, 30))
        self.label_36.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_36.setObjectName("label_36")
        self.gridLayout_6.addWidget(self.label_36, 0, 0, 1, 1)
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_7.setVerticalSpacing(15)
        self.formLayout_7.setObjectName("formLayout_7")
        spacerItem3 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_7.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem3)
        self.soilCoverLabel = QtWidgets.QLabel(self.soilMgmtWidget_base)
        self.soilCoverLabel.setObjectName("soilCoverLabel")
        self.formLayout_7.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.soilCoverLabel
        )
        self.verticalLayout_26 = QtWidgets.QVBoxLayout()
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.allCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        self.allCover.setObjectName("allCover")
        self.verticalLayout_26.addWidget(self.allCover)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setSpacing(8)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setSpacing(4)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.janCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.janCover.sizePolicy().hasHeightForWidth())
        self.janCover.setSizePolicy(sizePolicy)
        self.janCover.setAutoExclusive(False)
        self.janCover.setObjectName("janCover")
        self.verticalLayout_14.addWidget(self.janCover)
        self.febCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.febCover.sizePolicy().hasHeightForWidth())
        self.febCover.setSizePolicy(sizePolicy)
        self.febCover.setObjectName("febCover")
        self.verticalLayout_14.addWidget(self.febCover)
        self.marCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.marCover.sizePolicy().hasHeightForWidth())
        self.marCover.setSizePolicy(sizePolicy)
        self.marCover.setObjectName("marCover")
        self.verticalLayout_14.addWidget(self.marCover)
        self.aprCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aprCover.sizePolicy().hasHeightForWidth())
        self.aprCover.setSizePolicy(sizePolicy)
        self.aprCover.setObjectName("aprCover")
        self.verticalLayout_14.addWidget(self.aprCover)
        self.horizontalLayout_17.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setSpacing(4)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.mayCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mayCover.sizePolicy().hasHeightForWidth())
        self.mayCover.setSizePolicy(sizePolicy)
        self.mayCover.setObjectName("mayCover")
        self.verticalLayout_15.addWidget(self.mayCover)
        self.junCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.junCover.sizePolicy().hasHeightForWidth())
        self.junCover.setSizePolicy(sizePolicy)
        self.junCover.setObjectName("junCover")
        self.verticalLayout_15.addWidget(self.junCover)
        self.julCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.julCover.sizePolicy().hasHeightForWidth())
        self.julCover.setSizePolicy(sizePolicy)
        self.julCover.setObjectName("julCover")
        self.verticalLayout_15.addWidget(self.julCover)
        self.augCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.augCover.sizePolicy().hasHeightForWidth())
        self.augCover.setSizePolicy(sizePolicy)
        self.augCover.setObjectName("augCover")
        self.verticalLayout_15.addWidget(self.augCover)
        self.horizontalLayout_17.addLayout(self.verticalLayout_15)
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setSpacing(4)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.sepCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sepCover.sizePolicy().hasHeightForWidth())
        self.sepCover.setSizePolicy(sizePolicy)
        self.sepCover.setObjectName("sepCover")
        self.verticalLayout_16.addWidget(self.sepCover)
        self.octCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.octCover.sizePolicy().hasHeightForWidth())
        self.octCover.setSizePolicy(sizePolicy)
        self.octCover.setObjectName("octCover")
        self.verticalLayout_16.addWidget(self.octCover)
        self.novCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.novCover.sizePolicy().hasHeightForWidth())
        self.novCover.setSizePolicy(sizePolicy)
        self.novCover.setObjectName("novCover")
        self.verticalLayout_16.addWidget(self.novCover)
        self.decCover = QtWidgets.QCheckBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decCover.sizePolicy().hasHeightForWidth())
        self.decCover.setSizePolicy(sizePolicy)
        self.decCover.setObjectName("decCover")
        self.verticalLayout_16.addWidget(self.decCover)
        self.horizontalLayout_17.addLayout(self.verticalLayout_16)
        self.verticalLayout_26.addLayout(self.horizontalLayout_17)
        self.formLayout_7.setLayout(
            1, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_26
        )
        self.label_39 = QtWidgets.QLabel(self.soilMgmtWidget_base)
        self.label_39.setObjectName("label_39")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_39)
        self.verticalLayout_28 = QtWidgets.QVBoxLayout()
        self.verticalLayout_28.setObjectName("verticalLayout_28")
        self.fireNever = QtWidgets.QRadioButton(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fireNever.sizePolicy().hasHeightForWidth())
        self.fireNever.setSizePolicy(sizePolicy)
        self.fireNever.setObjectName("fireNever")
        self.verticalLayout_28.addWidget(self.fireNever)
        self.horizontalLayout_38 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_38.setSpacing(8)
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.fireInterval = QtWidgets.QRadioButton(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fireInterval.sizePolicy().hasHeightForWidth())
        self.fireInterval.setSizePolicy(sizePolicy)
        self.fireInterval.setObjectName("fireInterval")
        self.horizontalLayout_38.addWidget(self.fireInterval)
        self.fireFreq = QtWidgets.QSpinBox(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fireFreq.sizePolicy().hasHeightForWidth())
        self.fireFreq.setSizePolicy(sizePolicy)
        self.fireFreq.setMinimumSize(QtCore.QSize(60, 0))
        self.fireFreq.setMinimum(1)
        self.fireFreq.setSingleStep(5)
        self.fireFreq.setProperty("value", 10)
        self.fireFreq.setObjectName("fireFreq")
        self.horizontalLayout_38.addWidget(self.fireFreq)
        self.fireFreqLabel = QtWidgets.QLabel(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fireFreqLabel.sizePolicy().hasHeightForWidth()
        )
        self.fireFreqLabel.setSizePolicy(sizePolicy)
        self.fireFreqLabel.setObjectName("fireFreqLabel")
        self.horizontalLayout_38.addWidget(self.fireFreqLabel)
        self.verticalLayout_28.addLayout(self.horizontalLayout_38)
        self.fireCustom = QtWidgets.QRadioButton(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fireCustom.sizePolicy().hasHeightForWidth())
        self.fireCustom.setSizePolicy(sizePolicy)
        self.fireCustom.setObjectName("fireCustom")
        self.verticalLayout_28.addWidget(self.fireCustom)
        self.formLayout_7.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_28
        )
        self.fireCustomText = QtWidgets.QLabel(self.soilMgmtWidget_base)
        self.fireCustomText.setObjectName("fireCustomText")
        self.formLayout_7.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.fireCustomText
        )
        self.fireCustomLine = QtWidgets.QLineEdit(self.soilMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fireCustomLine.sizePolicy().hasHeightForWidth()
        )
        self.fireCustomLine.setSizePolicy(sizePolicy)
        self.fireCustomLine.setMinimumSize(QtCore.QSize(270, 0))
        self.fireCustomLine.setMaximumSize(QtCore.QSize(270, 16777215))
        self.fireCustomLine.setObjectName("fireCustomLine")
        self.formLayout_7.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.fireCustomLine
        )
        self.gridLayout_6.addLayout(self.formLayout_7, 1, 0, 1, 1)
        self.verticalLayout_34.addWidget(self.soilMgmtWidget_base)
        self.stackedWidget.addWidget(self.soilMgmtPage)
        self.cropMgmtPage_0 = QtWidgets.QWidget()
        self.cropMgmtPage_0.setObjectName("cropMgmtPage_0")
        self.verticalLayout_35 = QtWidgets.QVBoxLayout(self.cropMgmtPage_0)
        self.verticalLayout_35.setObjectName("verticalLayout_35")
        self.cropMgmtWidget_base = QtWidgets.QWidget(self.cropMgmtPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base.setObjectName("cropMgmtWidget_base")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.cropMgmtWidget_base)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_46 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_46.sizePolicy().hasHeightForWidth())
        self.label_46.setSizePolicy(sizePolicy)
        self.label_46.setMinimumSize(QtCore.QSize(0, 30))
        self.label_46.setObjectName("label_46")
        self.gridLayout_3.addWidget(self.label_46, 0, 0, 1, 1)
        self.formLayout_8 = QtWidgets.QFormLayout()
        self.formLayout_8.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_8.setVerticalSpacing(15)
        self.formLayout_8.setObjectName("formLayout_8")
        spacerItem4 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_8.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem4)
        self.label_45 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.label_45.setObjectName("label_45")
        self.formLayout_8.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_45)
        self.cropType_0 = QtWidgets.QComboBox(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_0.sizePolicy().hasHeightForWidth())
        self.cropType_0.setSizePolicy(sizePolicy)
        self.cropType_0.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_0.setObjectName("cropType_0")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.setItemText(1, "")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.setItemText(19, "")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.cropType_0.setItemText(29, "")
        self.cropType_0.addItem("")
        self.cropType_0.addItem("")
        self.formLayout_8.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cropType_0)
        self.cropInputType_0 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.cropInputType_0.setEnabled(False)
        self.cropInputType_0.setObjectName("cropInputType_0")
        self.formLayout_8.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_0
        )
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setSpacing(8)
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.cropCsvButton_0 = QtWidgets.QRadioButton(self.cropMgmtWidget_base)
        self.cropCsvButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_0.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_0.setSizePolicy(sizePolicy)
        self.cropCsvButton_0.setObjectName("cropCsvButton_0")
        self.verticalLayout_21.addWidget(self.cropCsvButton_0)
        self.cropCustomButton_0 = QtWidgets.QRadioButton(self.cropMgmtWidget_base)
        self.cropCustomButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_0.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_0.setSizePolicy(sizePolicy)
        self.cropCustomButton_0.setObjectName("cropCustomButton_0")
        self.verticalLayout_21.addWidget(self.cropCustomButton_0)
        self.formLayout_8.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_21
        )
        self.cropCsvLabel_0 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.cropCsvLabel_0.setEnabled(False)
        self.cropCsvLabel_0.setObjectName("cropCsvLabel_0")
        self.formLayout_8.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_0
        )
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setSpacing(8)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.cropCsvBrowseBox_0 = QtWidgets.QLineEdit(self.cropMgmtWidget_base)
        self.cropCsvBrowseBox_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_0.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_0.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_0.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_0.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_0.setMaxLength(1000)
        self.cropCsvBrowseBox_0.setObjectName("cropCsvBrowseBox_0")
        self.horizontalLayout_14.addWidget(self.cropCsvBrowseBox_0)
        self.cropCsvBrowseButton_0 = QtWidgets.QPushButton(self.cropMgmtWidget_base)
        self.cropCsvBrowseButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_0.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_0.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_0.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_0.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_0.setObjectName("cropCsvBrowseButton_0")
        self.horizontalLayout_14.addWidget(self.cropCsvBrowseButton_0)
        self.formLayout_8.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_14
        )
        self.label_44 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.label_44.setEnabled(True)
        self.label_44.setObjectName("label_44")
        self.formLayout_8.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_44)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setSpacing(8)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.cropYield_0 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_0.sizePolicy().hasHeightForWidth())
        self.cropYield_0.setSizePolicy(sizePolicy)
        self.cropYield_0.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_0.setDecimals(1)
        self.cropYield_0.setMaximum(20.0)
        self.cropYield_0.setSingleStep(0.5)
        self.cropYield_0.setProperty("value", 1.0)
        self.cropYield_0.setObjectName("cropYield_0")
        self.horizontalLayout_22.addWidget(self.cropYield_0)
        self.label_47 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_47.sizePolicy().hasHeightForWidth())
        self.label_47.setSizePolicy(sizePolicy)
        self.label_47.setObjectName("label_47")
        self.horizontalLayout_22.addWidget(self.label_47)
        self.formLayout_8.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_22
        )
        self.label_48 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.label_48.setObjectName("label_48")
        self.formLayout_8.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_48)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setSpacing(8)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.leftInField_0 = QtWidgets.QSpinBox(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_0.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_0.setSizePolicy(sizePolicy)
        self.leftInField_0.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_0.setMaximum(100)
        self.leftInField_0.setSingleStep(10)
        self.leftInField_0.setProperty("value", 100)
        self.leftInField_0.setObjectName("leftInField_0")
        self.horizontalLayout_23.addWidget(self.leftInField_0)
        self.label_49 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy)
        self.label_49.setObjectName("label_49")
        self.horizontalLayout_23.addWidget(self.label_49)
        self.formLayout_8.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_23
        )
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_36.setSpacing(8)
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.cropBurnYes_0 = QtWidgets.QRadioButton(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_0.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_0.setSizePolicy(sizePolicy)
        self.cropBurnYes_0.setObjectName("cropBurnYes_0")
        self.horizontalLayout_36.addWidget(self.cropBurnYes_0)
        self.cropBurnNo_0 = QtWidgets.QRadioButton(self.cropMgmtWidget_base)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_0.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_0.setSizePolicy(sizePolicy)
        self.cropBurnNo_0.setObjectName("cropBurnNo_0")
        self.horizontalLayout_36.addWidget(self.cropBurnNo_0)
        self.formLayout_8.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_36
        )
        self.label_2 = QtWidgets.QLabel(self.cropMgmtWidget_base)
        self.label_2.setObjectName("label_2")
        self.formLayout_8.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.gridLayout_3.addLayout(self.formLayout_8, 1, 0, 1, 1)
        self.verticalLayout_35.addWidget(self.cropMgmtWidget_base)
        self.stackedWidget.addWidget(self.cropMgmtPage_0)
        self.cropMgmtPage_1 = QtWidgets.QWidget()
        self.cropMgmtPage_1.setObjectName("cropMgmtPage_1")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.cropMgmtPage_1)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.cropMgmtWidget_base_2 = QtWidgets.QWidget(self.cropMgmtPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_2.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_2.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_2.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_2.setObjectName("cropMgmtWidget_base_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_60 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_60.sizePolicy().hasHeightForWidth())
        self.label_60.setSizePolicy(sizePolicy)
        self.label_60.setMinimumSize(QtCore.QSize(0, 30))
        self.label_60.setObjectName("label_60")
        self.gridLayout_5.addWidget(self.label_60, 0, 0, 1, 1)
        self.formLayout_11 = QtWidgets.QFormLayout()
        self.formLayout_11.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_11.setVerticalSpacing(15)
        self.formLayout_11.setObjectName("formLayout_11")
        spacerItem5 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_11.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem5)
        self.label_61 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.label_61.setObjectName("label_61")
        self.formLayout_11.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_61)
        self.cropType_1 = QtWidgets.QComboBox(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_1.sizePolicy().hasHeightForWidth())
        self.cropType_1.setSizePolicy(sizePolicy)
        self.cropType_1.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_1.setObjectName("cropType_1")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.setItemText(1, "")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.setItemText(19, "")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.cropType_1.setItemText(29, "")
        self.cropType_1.addItem("")
        self.cropType_1.addItem("")
        self.formLayout_11.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_1
        )
        self.cropInputType_1 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.cropInputType_1.setEnabled(False)
        self.cropInputType_1.setObjectName("cropInputType_1")
        self.formLayout_11.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_1
        )
        self.verticalLayout_22 = QtWidgets.QVBoxLayout()
        self.verticalLayout_22.setSpacing(8)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.cropCsvButton_1 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_2)
        self.cropCsvButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_1.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_1.setSizePolicy(sizePolicy)
        self.cropCsvButton_1.setObjectName("cropCsvButton_1")
        self.verticalLayout_22.addWidget(self.cropCsvButton_1)
        self.cropCustomButton_1 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_2)
        self.cropCustomButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_1.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_1.setSizePolicy(sizePolicy)
        self.cropCustomButton_1.setObjectName("cropCustomButton_1")
        self.verticalLayout_22.addWidget(self.cropCustomButton_1)
        self.formLayout_11.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_22
        )
        self.cropCsvLabel_1 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.cropCsvLabel_1.setEnabled(False)
        self.cropCsvLabel_1.setObjectName("cropCsvLabel_1")
        self.formLayout_11.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_1
        )
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setSpacing(8)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.cropCsvBrowseBox_1 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_2)
        self.cropCsvBrowseBox_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_1.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_1.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_1.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_1.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_1.setMaxLength(1000)
        self.cropCsvBrowseBox_1.setObjectName("cropCsvBrowseBox_1")
        self.horizontalLayout_18.addWidget(self.cropCsvBrowseBox_1)
        self.cropCsvBrowseButton_1 = QtWidgets.QPushButton(self.cropMgmtWidget_base_2)
        self.cropCsvBrowseButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_1.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_1.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_1.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_1.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_1.setObjectName("cropCsvBrowseButton_1")
        self.horizontalLayout_18.addWidget(self.cropCsvBrowseButton_1)
        self.formLayout_11.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_18
        )
        self.label_63 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.label_63.setEnabled(True)
        self.label_63.setObjectName("label_63")
        self.formLayout_11.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_63)
        self.horizontalLayout_41 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_41.setSpacing(8)
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.cropYield_1 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_1.sizePolicy().hasHeightForWidth())
        self.cropYield_1.setSizePolicy(sizePolicy)
        self.cropYield_1.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_1.setDecimals(1)
        self.cropYield_1.setMaximum(20.0)
        self.cropYield_1.setSingleStep(0.5)
        self.cropYield_1.setProperty("value", 1.0)
        self.cropYield_1.setObjectName("cropYield_1")
        self.horizontalLayout_41.addWidget(self.cropYield_1)
        self.label_64 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_64.sizePolicy().hasHeightForWidth())
        self.label_64.setSizePolicy(sizePolicy)
        self.label_64.setObjectName("label_64")
        self.horizontalLayout_41.addWidget(self.label_64)
        self.formLayout_11.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_41
        )
        self.label_65 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.label_65.setObjectName("label_65")
        self.formLayout_11.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_65)
        self.horizontalLayout_42 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_42.setSpacing(8)
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.leftInField_1 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_1.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_1.setSizePolicy(sizePolicy)
        self.leftInField_1.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_1.setMaximum(100)
        self.leftInField_1.setSingleStep(10)
        self.leftInField_1.setProperty("value", 100)
        self.leftInField_1.setObjectName("leftInField_1")
        self.horizontalLayout_42.addWidget(self.leftInField_1)
        self.label_68 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_68.sizePolicy().hasHeightForWidth())
        self.label_68.setSizePolicy(sizePolicy)
        self.label_68.setObjectName("label_68")
        self.horizontalLayout_42.addWidget(self.label_68)
        self.formLayout_11.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_42
        )
        self.horizontalLayout_43 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_43.setSpacing(8)
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.cropBurnYes_1 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_1.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_1.setSizePolicy(sizePolicy)
        self.cropBurnYes_1.setObjectName("cropBurnYes_1")
        self.horizontalLayout_43.addWidget(self.cropBurnYes_1)
        self.cropBurnNo_1 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_1.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_1.setSizePolicy(sizePolicy)
        self.cropBurnNo_1.setObjectName("cropBurnNo_1")
        self.horizontalLayout_43.addWidget(self.cropBurnNo_1)
        self.formLayout_11.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_43
        )
        self.label_3 = QtWidgets.QLabel(self.cropMgmtWidget_base_2)
        self.label_3.setObjectName("label_3")
        self.formLayout_11.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.gridLayout_5.addLayout(self.formLayout_11, 1, 0, 1, 1)
        self.verticalLayout_13.addWidget(self.cropMgmtWidget_base_2)
        self.stackedWidget.addWidget(self.cropMgmtPage_1)
        self.cropMgmtPage_2 = QtWidgets.QWidget()
        self.cropMgmtPage_2.setObjectName("cropMgmtPage_2")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.cropMgmtPage_2)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.cropMgmtWidget_base_3 = QtWidgets.QWidget(self.cropMgmtPage_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_3.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_3.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_3.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_3.setObjectName("cropMgmtWidget_base_3")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_3)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_69 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_69.sizePolicy().hasHeightForWidth())
        self.label_69.setSizePolicy(sizePolicy)
        self.label_69.setMinimumSize(QtCore.QSize(0, 30))
        self.label_69.setObjectName("label_69")
        self.gridLayout_9.addWidget(self.label_69, 0, 0, 1, 1)
        self.formLayout_16 = QtWidgets.QFormLayout()
        self.formLayout_16.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_16.setVerticalSpacing(15)
        self.formLayout_16.setObjectName("formLayout_16")
        spacerItem6 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_16.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem6)
        self.label_70 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.label_70.setObjectName("label_70")
        self.formLayout_16.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_70)
        self.cropType_2 = QtWidgets.QComboBox(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_2.sizePolicy().hasHeightForWidth())
        self.cropType_2.setSizePolicy(sizePolicy)
        self.cropType_2.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_2.setObjectName("cropType_2")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.setItemText(1, "")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.setItemText(19, "")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.cropType_2.setItemText(29, "")
        self.cropType_2.addItem("")
        self.cropType_2.addItem("")
        self.formLayout_16.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_2
        )
        self.cropInputType_2 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.cropInputType_2.setEnabled(False)
        self.cropInputType_2.setObjectName("cropInputType_2")
        self.formLayout_16.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_2
        )
        self.verticalLayout_24 = QtWidgets.QVBoxLayout()
        self.verticalLayout_24.setSpacing(8)
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.cropCsvButton_2 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_3)
        self.cropCsvButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_2.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_2.setSizePolicy(sizePolicy)
        self.cropCsvButton_2.setObjectName("cropCsvButton_2")
        self.verticalLayout_24.addWidget(self.cropCsvButton_2)
        self.cropCustomButton_2 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_3)
        self.cropCustomButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_2.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_2.setSizePolicy(sizePolicy)
        self.cropCustomButton_2.setObjectName("cropCustomButton_2")
        self.verticalLayout_24.addWidget(self.cropCustomButton_2)
        self.formLayout_16.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_24
        )
        self.cropCsvLabel_2 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.cropCsvLabel_2.setEnabled(False)
        self.cropCsvLabel_2.setObjectName("cropCsvLabel_2")
        self.formLayout_16.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_2
        )
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setSpacing(8)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.cropCsvBrowseBox_2 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_3)
        self.cropCsvBrowseBox_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_2.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_2.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_2.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_2.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_2.setMaxLength(1000)
        self.cropCsvBrowseBox_2.setObjectName("cropCsvBrowseBox_2")
        self.horizontalLayout_21.addWidget(self.cropCsvBrowseBox_2)
        self.cropCsvBrowseButton_2 = QtWidgets.QPushButton(self.cropMgmtWidget_base_3)
        self.cropCsvBrowseButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_2.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_2.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_2.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_2.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_2.setObjectName("cropCsvBrowseButton_2")
        self.horizontalLayout_21.addWidget(self.cropCsvBrowseButton_2)
        self.formLayout_16.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_21
        )
        self.label_71 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.label_71.setEnabled(True)
        self.label_71.setObjectName("label_71")
        self.formLayout_16.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_71)
        self.horizontalLayout_44 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_44.setSpacing(8)
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.cropYield_2 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_2.sizePolicy().hasHeightForWidth())
        self.cropYield_2.setSizePolicy(sizePolicy)
        self.cropYield_2.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_2.setDecimals(1)
        self.cropYield_2.setMaximum(20.0)
        self.cropYield_2.setSingleStep(0.5)
        self.cropYield_2.setProperty("value", 1.0)
        self.cropYield_2.setObjectName("cropYield_2")
        self.horizontalLayout_44.addWidget(self.cropYield_2)
        self.label_72 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_72.sizePolicy().hasHeightForWidth())
        self.label_72.setSizePolicy(sizePolicy)
        self.label_72.setObjectName("label_72")
        self.horizontalLayout_44.addWidget(self.label_72)
        self.formLayout_16.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_44
        )
        self.label_73 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.label_73.setObjectName("label_73")
        self.formLayout_16.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_73)
        self.horizontalLayout_46 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_46.setSpacing(8)
        self.horizontalLayout_46.setObjectName("horizontalLayout_46")
        self.cropBurnYes_2 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_2.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_2.setSizePolicy(sizePolicy)
        self.cropBurnYes_2.setObjectName("cropBurnYes_2")
        self.horizontalLayout_46.addWidget(self.cropBurnYes_2)
        self.cropBurnNo_2 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_2.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_2.setSizePolicy(sizePolicy)
        self.cropBurnNo_2.setObjectName("cropBurnNo_2")
        self.horizontalLayout_46.addWidget(self.cropBurnNo_2)
        self.formLayout_16.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_46
        )
        self.label_4 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        self.label_4.setObjectName("label_4")
        self.formLayout_16.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_45.setSpacing(8)
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.leftInField_2 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_2.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_2.setSizePolicy(sizePolicy)
        self.leftInField_2.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_2.setMaximum(100)
        self.leftInField_2.setSingleStep(10)
        self.leftInField_2.setProperty("value", 100)
        self.leftInField_2.setObjectName("leftInField_2")
        self.horizontalLayout_45.addWidget(self.leftInField_2)
        self.label_74 = QtWidgets.QLabel(self.cropMgmtWidget_base_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_74.sizePolicy().hasHeightForWidth())
        self.label_74.setSizePolicy(sizePolicy)
        self.label_74.setObjectName("label_74")
        self.horizontalLayout_45.addWidget(self.label_74)
        self.formLayout_16.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_45
        )
        self.gridLayout_9.addLayout(self.formLayout_16, 1, 0, 1, 1)
        self.verticalLayout_17.addWidget(self.cropMgmtWidget_base_3)
        self.stackedWidget.addWidget(self.cropMgmtPage_2)
        self.cropMgmtPage_3 = QtWidgets.QWidget()
        self.cropMgmtPage_3.setObjectName("cropMgmtPage_3")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.cropMgmtPage_3)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.cropMgmtWidget_base_4 = QtWidgets.QWidget(self.cropMgmtPage_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_4.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_4.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_4.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_4.setObjectName("cropMgmtWidget_base_4")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_4)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_75 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_75.sizePolicy().hasHeightForWidth())
        self.label_75.setSizePolicy(sizePolicy)
        self.label_75.setMinimumSize(QtCore.QSize(0, 30))
        self.label_75.setObjectName("label_75")
        self.gridLayout_10.addWidget(self.label_75, 0, 0, 1, 1)
        self.formLayout_17 = QtWidgets.QFormLayout()
        self.formLayout_17.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_17.setVerticalSpacing(15)
        self.formLayout_17.setObjectName("formLayout_17")
        spacerItem7 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_17.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem7)
        self.label_76 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.label_76.setObjectName("label_76")
        self.formLayout_17.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_76)
        self.cropType_3 = QtWidgets.QComboBox(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_3.sizePolicy().hasHeightForWidth())
        self.cropType_3.setSizePolicy(sizePolicy)
        self.cropType_3.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_3.setObjectName("cropType_3")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.setItemText(1, "")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.setItemText(19, "")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.cropType_3.setItemText(29, "")
        self.cropType_3.addItem("")
        self.cropType_3.addItem("")
        self.formLayout_17.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_3
        )
        self.cropInputType_3 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.cropInputType_3.setEnabled(False)
        self.cropInputType_3.setObjectName("cropInputType_3")
        self.formLayout_17.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_3
        )
        self.verticalLayout_25 = QtWidgets.QVBoxLayout()
        self.verticalLayout_25.setSpacing(8)
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.cropCsvButton_3 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_4)
        self.cropCsvButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_3.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_3.setSizePolicy(sizePolicy)
        self.cropCsvButton_3.setObjectName("cropCsvButton_3")
        self.verticalLayout_25.addWidget(self.cropCsvButton_3)
        self.cropCustomButton_3 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_4)
        self.cropCustomButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_3.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_3.setSizePolicy(sizePolicy)
        self.cropCustomButton_3.setObjectName("cropCustomButton_3")
        self.verticalLayout_25.addWidget(self.cropCustomButton_3)
        self.formLayout_17.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_25
        )
        self.cropCsvLabel_3 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.cropCsvLabel_3.setEnabled(False)
        self.cropCsvLabel_3.setObjectName("cropCsvLabel_3")
        self.formLayout_17.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_3
        )
        self.horizontalLayout_47 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_47.setSpacing(8)
        self.horizontalLayout_47.setObjectName("horizontalLayout_47")
        self.cropCsvBrowseBox_3 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_4)
        self.cropCsvBrowseBox_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_3.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_3.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_3.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_3.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_3.setMaxLength(1000)
        self.cropCsvBrowseBox_3.setObjectName("cropCsvBrowseBox_3")
        self.horizontalLayout_47.addWidget(self.cropCsvBrowseBox_3)
        self.cropCsvBrowseButton_3 = QtWidgets.QPushButton(self.cropMgmtWidget_base_4)
        self.cropCsvBrowseButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_3.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_3.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_3.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_3.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_3.setObjectName("cropCsvBrowseButton_3")
        self.horizontalLayout_47.addWidget(self.cropCsvBrowseButton_3)
        self.formLayout_17.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_47
        )
        self.label_77 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.label_77.setEnabled(True)
        self.label_77.setObjectName("label_77")
        self.formLayout_17.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_77)
        self.horizontalLayout_48 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_48.setSpacing(8)
        self.horizontalLayout_48.setObjectName("horizontalLayout_48")
        self.cropYield_3 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_3.sizePolicy().hasHeightForWidth())
        self.cropYield_3.setSizePolicy(sizePolicy)
        self.cropYield_3.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_3.setDecimals(1)
        self.cropYield_3.setMaximum(20.0)
        self.cropYield_3.setSingleStep(0.5)
        self.cropYield_3.setProperty("value", 1.0)
        self.cropYield_3.setObjectName("cropYield_3")
        self.horizontalLayout_48.addWidget(self.cropYield_3)
        self.label_78 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_78.sizePolicy().hasHeightForWidth())
        self.label_78.setSizePolicy(sizePolicy)
        self.label_78.setObjectName("label_78")
        self.horizontalLayout_48.addWidget(self.label_78)
        self.formLayout_17.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_48
        )
        self.label_79 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.label_79.setObjectName("label_79")
        self.formLayout_17.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_79)
        self.horizontalLayout_49 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_49.setSpacing(8)
        self.horizontalLayout_49.setObjectName("horizontalLayout_49")
        self.leftInField_3 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_3.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_3.setSizePolicy(sizePolicy)
        self.leftInField_3.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_3.setMaximum(100)
        self.leftInField_3.setSingleStep(10)
        self.leftInField_3.setProperty("value", 100)
        self.leftInField_3.setObjectName("leftInField_3")
        self.horizontalLayout_49.addWidget(self.leftInField_3)
        self.label_80 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_80.sizePolicy().hasHeightForWidth())
        self.label_80.setSizePolicy(sizePolicy)
        self.label_80.setObjectName("label_80")
        self.horizontalLayout_49.addWidget(self.label_80)
        self.formLayout_17.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_49
        )
        self.horizontalLayout_50 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_50.setSpacing(8)
        self.horizontalLayout_50.setObjectName("horizontalLayout_50")
        self.cropBurnYes_3 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_3.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_3.setSizePolicy(sizePolicy)
        self.cropBurnYes_3.setObjectName("cropBurnYes_3")
        self.horizontalLayout_50.addWidget(self.cropBurnYes_3)
        self.cropBurnNo_3 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_3.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_3.setSizePolicy(sizePolicy)
        self.cropBurnNo_3.setObjectName("cropBurnNo_3")
        self.horizontalLayout_50.addWidget(self.cropBurnNo_3)
        self.formLayout_17.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_50
        )
        self.label_5 = QtWidgets.QLabel(self.cropMgmtWidget_base_4)
        self.label_5.setObjectName("label_5")
        self.formLayout_17.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.gridLayout_10.addLayout(self.formLayout_17, 1, 0, 1, 1)
        self.verticalLayout_18.addWidget(self.cropMgmtWidget_base_4)
        self.stackedWidget.addWidget(self.cropMgmtPage_3)
        self.cropMgmtPage_4 = QtWidgets.QWidget()
        self.cropMgmtPage_4.setObjectName("cropMgmtPage_4")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.cropMgmtPage_4)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.cropMgmtWidget_base_5 = QtWidgets.QWidget(self.cropMgmtPage_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_5.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_5.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_5.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_5.setObjectName("cropMgmtWidget_base_5")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_5)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_81 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_81.sizePolicy().hasHeightForWidth())
        self.label_81.setSizePolicy(sizePolicy)
        self.label_81.setMinimumSize(QtCore.QSize(0, 30))
        self.label_81.setObjectName("label_81")
        self.gridLayout_11.addWidget(self.label_81, 0, 0, 1, 1)
        self.formLayout_18 = QtWidgets.QFormLayout()
        self.formLayout_18.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_18.setVerticalSpacing(15)
        self.formLayout_18.setObjectName("formLayout_18")
        spacerItem8 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_18.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem8)
        self.label_82 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.label_82.setObjectName("label_82")
        self.formLayout_18.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_82)
        self.cropType_4 = QtWidgets.QComboBox(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_4.sizePolicy().hasHeightForWidth())
        self.cropType_4.setSizePolicy(sizePolicy)
        self.cropType_4.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_4.setObjectName("cropType_4")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.setItemText(1, "")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.setItemText(19, "")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.cropType_4.setItemText(29, "")
        self.cropType_4.addItem("")
        self.cropType_4.addItem("")
        self.formLayout_18.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_4
        )
        self.cropInputType_4 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.cropInputType_4.setEnabled(False)
        self.cropInputType_4.setObjectName("cropInputType_4")
        self.formLayout_18.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_4
        )
        self.verticalLayout_30 = QtWidgets.QVBoxLayout()
        self.verticalLayout_30.setSpacing(8)
        self.verticalLayout_30.setObjectName("verticalLayout_30")
        self.cropCsvButton_4 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_5)
        self.cropCsvButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_4.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_4.setSizePolicy(sizePolicy)
        self.cropCsvButton_4.setObjectName("cropCsvButton_4")
        self.verticalLayout_30.addWidget(self.cropCsvButton_4)
        self.cropCustomButton_4 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_5)
        self.cropCustomButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_4.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_4.setSizePolicy(sizePolicy)
        self.cropCustomButton_4.setObjectName("cropCustomButton_4")
        self.verticalLayout_30.addWidget(self.cropCustomButton_4)
        self.formLayout_18.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_30
        )
        self.cropCsvLabel_4 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.cropCsvLabel_4.setEnabled(False)
        self.cropCsvLabel_4.setObjectName("cropCsvLabel_4")
        self.formLayout_18.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_4
        )
        self.horizontalLayout_51 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_51.setSpacing(8)
        self.horizontalLayout_51.setObjectName("horizontalLayout_51")
        self.cropCsvBrowseBox_4 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_5)
        self.cropCsvBrowseBox_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_4.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_4.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_4.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_4.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_4.setMaxLength(1000)
        self.cropCsvBrowseBox_4.setObjectName("cropCsvBrowseBox_4")
        self.horizontalLayout_51.addWidget(self.cropCsvBrowseBox_4)
        self.cropCsvBrowseButton_4 = QtWidgets.QPushButton(self.cropMgmtWidget_base_5)
        self.cropCsvBrowseButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_4.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_4.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_4.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_4.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_4.setObjectName("cropCsvBrowseButton_4")
        self.horizontalLayout_51.addWidget(self.cropCsvBrowseButton_4)
        self.formLayout_18.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_51
        )
        self.label_83 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.label_83.setEnabled(True)
        self.label_83.setObjectName("label_83")
        self.formLayout_18.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_83)
        self.horizontalLayout_52 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_52.setSpacing(8)
        self.horizontalLayout_52.setObjectName("horizontalLayout_52")
        self.cropYield_4 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_4.sizePolicy().hasHeightForWidth())
        self.cropYield_4.setSizePolicy(sizePolicy)
        self.cropYield_4.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_4.setDecimals(1)
        self.cropYield_4.setMaximum(20.0)
        self.cropYield_4.setSingleStep(0.5)
        self.cropYield_4.setProperty("value", 1.0)
        self.cropYield_4.setObjectName("cropYield_4")
        self.horizontalLayout_52.addWidget(self.cropYield_4)
        self.label_84 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_84.sizePolicy().hasHeightForWidth())
        self.label_84.setSizePolicy(sizePolicy)
        self.label_84.setObjectName("label_84")
        self.horizontalLayout_52.addWidget(self.label_84)
        self.formLayout_18.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_52
        )
        self.label_85 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.label_85.setObjectName("label_85")
        self.formLayout_18.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_85)
        self.horizontalLayout_53 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_53.setSpacing(8)
        self.horizontalLayout_53.setObjectName("horizontalLayout_53")
        self.leftInField_4 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_4.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_4.setSizePolicy(sizePolicy)
        self.leftInField_4.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_4.setMaximum(100)
        self.leftInField_4.setSingleStep(10)
        self.leftInField_4.setProperty("value", 100)
        self.leftInField_4.setObjectName("leftInField_4")
        self.horizontalLayout_53.addWidget(self.leftInField_4)
        self.label_86 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_86.sizePolicy().hasHeightForWidth())
        self.label_86.setSizePolicy(sizePolicy)
        self.label_86.setObjectName("label_86")
        self.horizontalLayout_53.addWidget(self.label_86)
        self.formLayout_18.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_53
        )
        self.horizontalLayout_54 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_54.setSpacing(8)
        self.horizontalLayout_54.setObjectName("horizontalLayout_54")
        self.cropBurnYes_4 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_4.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_4.setSizePolicy(sizePolicy)
        self.cropBurnYes_4.setObjectName("cropBurnYes_4")
        self.horizontalLayout_54.addWidget(self.cropBurnYes_4)
        self.cropBurnNo_4 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_4.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_4.setSizePolicy(sizePolicy)
        self.cropBurnNo_4.setObjectName("cropBurnNo_4")
        self.horizontalLayout_54.addWidget(self.cropBurnNo_4)
        self.formLayout_18.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_54
        )
        self.label_7 = QtWidgets.QLabel(self.cropMgmtWidget_base_5)
        self.label_7.setObjectName("label_7")
        self.formLayout_18.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.gridLayout_11.addLayout(self.formLayout_18, 1, 0, 1, 1)
        self.verticalLayout_19.addWidget(self.cropMgmtWidget_base_5)
        self.stackedWidget.addWidget(self.cropMgmtPage_4)
        self.cropMgmtPage_5 = QtWidgets.QWidget()
        self.cropMgmtPage_5.setObjectName("cropMgmtPage_5")
        self.verticalLayout_71 = QtWidgets.QVBoxLayout(self.cropMgmtPage_5)
        self.verticalLayout_71.setObjectName("verticalLayout_71")
        self.cropMgmtWidget_base_6 = QtWidgets.QWidget(self.cropMgmtPage_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_6.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_6.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_6.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_6.setObjectName("cropMgmtWidget_base_6")
        self.gridLayout_30 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_6)
        self.gridLayout_30.setObjectName("gridLayout_30")
        self.label_129 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_129.sizePolicy().hasHeightForWidth())
        self.label_129.setSizePolicy(sizePolicy)
        self.label_129.setMinimumSize(QtCore.QSize(0, 30))
        self.label_129.setObjectName("label_129")
        self.gridLayout_30.addWidget(self.label_129, 0, 0, 1, 1)
        self.formLayout_37 = QtWidgets.QFormLayout()
        self.formLayout_37.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_37.setVerticalSpacing(15)
        self.formLayout_37.setObjectName("formLayout_37")
        spacerItem9 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_37.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem9)
        self.label_130 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.label_130.setObjectName("label_130")
        self.formLayout_37.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_130)
        self.cropType_5 = QtWidgets.QComboBox(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_5.sizePolicy().hasHeightForWidth())
        self.cropType_5.setSizePolicy(sizePolicy)
        self.cropType_5.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_5.setObjectName("cropType_5")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.setItemText(1, "")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.setItemText(19, "")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.cropType_5.setItemText(29, "")
        self.cropType_5.addItem("")
        self.cropType_5.addItem("")
        self.formLayout_37.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_5
        )
        self.cropInputType_5 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.cropInputType_5.setEnabled(False)
        self.cropInputType_5.setObjectName("cropInputType_5")
        self.formLayout_37.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_5
        )
        self.verticalLayout_70 = QtWidgets.QVBoxLayout()
        self.verticalLayout_70.setSpacing(8)
        self.verticalLayout_70.setObjectName("verticalLayout_70")
        self.cropCsvButton_5 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_6)
        self.cropCsvButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_5.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_5.setSizePolicy(sizePolicy)
        self.cropCsvButton_5.setObjectName("cropCsvButton_5")
        self.verticalLayout_70.addWidget(self.cropCsvButton_5)
        self.cropCustomButton_5 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_6)
        self.cropCustomButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_5.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_5.setSizePolicy(sizePolicy)
        self.cropCustomButton_5.setObjectName("cropCustomButton_5")
        self.verticalLayout_70.addWidget(self.cropCustomButton_5)
        self.formLayout_37.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_70
        )
        self.cropCsvLabel_5 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.cropCsvLabel_5.setEnabled(False)
        self.cropCsvLabel_5.setObjectName("cropCsvLabel_5")
        self.formLayout_37.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_5
        )
        self.horizontalLayout_63 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_63.setSpacing(8)
        self.horizontalLayout_63.setObjectName("horizontalLayout_63")
        self.cropCsvBrowseBox_5 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_6)
        self.cropCsvBrowseBox_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_5.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_5.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_5.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_5.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_5.setMaxLength(1000)
        self.cropCsvBrowseBox_5.setObjectName("cropCsvBrowseBox_5")
        self.horizontalLayout_63.addWidget(self.cropCsvBrowseBox_5)
        self.cropCsvBrowseButton_5 = QtWidgets.QPushButton(self.cropMgmtWidget_base_6)
        self.cropCsvBrowseButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_5.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_5.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_5.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_5.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_5.setObjectName("cropCsvBrowseButton_5")
        self.horizontalLayout_63.addWidget(self.cropCsvBrowseButton_5)
        self.formLayout_37.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_63
        )
        self.label_131 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.label_131.setEnabled(True)
        self.label_131.setObjectName("label_131")
        self.formLayout_37.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_131)
        self.horizontalLayout_75 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_75.setSpacing(8)
        self.horizontalLayout_75.setObjectName("horizontalLayout_75")
        self.cropYield_5 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_5.sizePolicy().hasHeightForWidth())
        self.cropYield_5.setSizePolicy(sizePolicy)
        self.cropYield_5.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_5.setDecimals(1)
        self.cropYield_5.setMaximum(20.0)
        self.cropYield_5.setSingleStep(0.5)
        self.cropYield_5.setProperty("value", 1.0)
        self.cropYield_5.setObjectName("cropYield_5")
        self.horizontalLayout_75.addWidget(self.cropYield_5)
        self.label_132 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_132.sizePolicy().hasHeightForWidth())
        self.label_132.setSizePolicy(sizePolicy)
        self.label_132.setObjectName("label_132")
        self.horizontalLayout_75.addWidget(self.label_132)
        self.formLayout_37.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_75
        )
        self.label_133 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.label_133.setObjectName("label_133")
        self.formLayout_37.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_133)
        self.horizontalLayout_87 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_87.setSpacing(8)
        self.horizontalLayout_87.setObjectName("horizontalLayout_87")
        self.leftInField_5 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_5.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_5.setSizePolicy(sizePolicy)
        self.leftInField_5.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_5.setMaximum(100)
        self.leftInField_5.setSingleStep(10)
        self.leftInField_5.setProperty("value", 100)
        self.leftInField_5.setObjectName("leftInField_5")
        self.horizontalLayout_87.addWidget(self.leftInField_5)
        self.label_134 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_134.sizePolicy().hasHeightForWidth())
        self.label_134.setSizePolicy(sizePolicy)
        self.label_134.setObjectName("label_134")
        self.horizontalLayout_87.addWidget(self.label_134)
        self.formLayout_37.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_87
        )
        self.horizontalLayout_99 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_99.setSpacing(8)
        self.horizontalLayout_99.setObjectName("horizontalLayout_99")
        self.cropBurnYes_5 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_5.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_5.setSizePolicy(sizePolicy)
        self.cropBurnYes_5.setObjectName("cropBurnYes_5")
        self.horizontalLayout_99.addWidget(self.cropBurnYes_5)
        self.cropBurnNo_5 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_5.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_5.setSizePolicy(sizePolicy)
        self.cropBurnNo_5.setObjectName("cropBurnNo_5")
        self.horizontalLayout_99.addWidget(self.cropBurnNo_5)
        self.formLayout_37.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_99
        )
        self.label_8 = QtWidgets.QLabel(self.cropMgmtWidget_base_6)
        self.label_8.setObjectName("label_8")
        self.formLayout_37.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.gridLayout_30.addLayout(self.formLayout_37, 1, 0, 1, 1)
        self.verticalLayout_71.addWidget(self.cropMgmtWidget_base_6)
        self.stackedWidget.addWidget(self.cropMgmtPage_5)
        self.cropMgmtPage_6 = QtWidgets.QWidget()
        self.cropMgmtPage_6.setObjectName("cropMgmtPage_6")
        self.verticalLayout_73 = QtWidgets.QVBoxLayout(self.cropMgmtPage_6)
        self.verticalLayout_73.setObjectName("verticalLayout_73")
        self.cropMgmtWidget_base_7 = QtWidgets.QWidget(self.cropMgmtPage_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_7.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_7.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_7.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_7.setObjectName("cropMgmtWidget_base_7")
        self.gridLayout_31 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_7)
        self.gridLayout_31.setObjectName("gridLayout_31")
        self.label_135 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_135.sizePolicy().hasHeightForWidth())
        self.label_135.setSizePolicy(sizePolicy)
        self.label_135.setMinimumSize(QtCore.QSize(0, 30))
        self.label_135.setObjectName("label_135")
        self.gridLayout_31.addWidget(self.label_135, 0, 0, 1, 1)
        self.formLayout_38 = QtWidgets.QFormLayout()
        self.formLayout_38.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_38.setVerticalSpacing(15)
        self.formLayout_38.setObjectName("formLayout_38")
        spacerItem10 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_38.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem10)
        self.label_136 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.label_136.setObjectName("label_136")
        self.formLayout_38.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_136)
        self.cropType_6 = QtWidgets.QComboBox(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_6.sizePolicy().hasHeightForWidth())
        self.cropType_6.setSizePolicy(sizePolicy)
        self.cropType_6.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_6.setObjectName("cropType_6")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.setItemText(1, "")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.setItemText(19, "")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.cropType_6.setItemText(29, "")
        self.cropType_6.addItem("")
        self.cropType_6.addItem("")
        self.formLayout_38.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_6
        )
        self.cropInputType_6 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.cropInputType_6.setEnabled(False)
        self.cropInputType_6.setObjectName("cropInputType_6")
        self.formLayout_38.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_6
        )
        self.verticalLayout_72 = QtWidgets.QVBoxLayout()
        self.verticalLayout_72.setSpacing(8)
        self.verticalLayout_72.setObjectName("verticalLayout_72")
        self.cropCsvButton_6 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_7)
        self.cropCsvButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_6.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_6.setSizePolicy(sizePolicy)
        self.cropCsvButton_6.setObjectName("cropCsvButton_6")
        self.verticalLayout_72.addWidget(self.cropCsvButton_6)
        self.cropCustomButton_6 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_7)
        self.cropCustomButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_6.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_6.setSizePolicy(sizePolicy)
        self.cropCustomButton_6.setObjectName("cropCustomButton_6")
        self.verticalLayout_72.addWidget(self.cropCustomButton_6)
        self.formLayout_38.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_72
        )
        self.cropCsvLabel_6 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.cropCsvLabel_6.setEnabled(False)
        self.cropCsvLabel_6.setObjectName("cropCsvLabel_6")
        self.formLayout_38.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_6
        )
        self.horizontalLayout_108 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_108.setSpacing(8)
        self.horizontalLayout_108.setObjectName("horizontalLayout_108")
        self.cropCsvBrowseBox_6 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_7)
        self.cropCsvBrowseBox_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_6.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_6.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_6.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_6.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_6.setMaxLength(1000)
        self.cropCsvBrowseBox_6.setObjectName("cropCsvBrowseBox_6")
        self.horizontalLayout_108.addWidget(self.cropCsvBrowseBox_6)
        self.cropCsvBrowseButton_6 = QtWidgets.QPushButton(self.cropMgmtWidget_base_7)
        self.cropCsvBrowseButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_6.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_6.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_6.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_6.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_6.setObjectName("cropCsvBrowseButton_6")
        self.horizontalLayout_108.addWidget(self.cropCsvBrowseButton_6)
        self.formLayout_38.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_108
        )
        self.label_137 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.label_137.setEnabled(True)
        self.label_137.setObjectName("label_137")
        self.formLayout_38.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_137)
        self.horizontalLayout_109 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_109.setSpacing(8)
        self.horizontalLayout_109.setObjectName("horizontalLayout_109")
        self.cropYield_6 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_6.sizePolicy().hasHeightForWidth())
        self.cropYield_6.setSizePolicy(sizePolicy)
        self.cropYield_6.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_6.setDecimals(1)
        self.cropYield_6.setMaximum(20.0)
        self.cropYield_6.setSingleStep(0.5)
        self.cropYield_6.setProperty("value", 1.0)
        self.cropYield_6.setObjectName("cropYield_6")
        self.horizontalLayout_109.addWidget(self.cropYield_6)
        self.label_138 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_138.sizePolicy().hasHeightForWidth())
        self.label_138.setSizePolicy(sizePolicy)
        self.label_138.setObjectName("label_138")
        self.horizontalLayout_109.addWidget(self.label_138)
        self.formLayout_38.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_109
        )
        self.label_139 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.label_139.setObjectName("label_139")
        self.formLayout_38.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_139)
        self.horizontalLayout_110 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_110.setSpacing(8)
        self.horizontalLayout_110.setObjectName("horizontalLayout_110")
        self.leftInField_6 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_6.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_6.setSizePolicy(sizePolicy)
        self.leftInField_6.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_6.setMaximum(100)
        self.leftInField_6.setSingleStep(10)
        self.leftInField_6.setProperty("value", 100)
        self.leftInField_6.setObjectName("leftInField_6")
        self.horizontalLayout_110.addWidget(self.leftInField_6)
        self.label_140 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_140.sizePolicy().hasHeightForWidth())
        self.label_140.setSizePolicy(sizePolicy)
        self.label_140.setObjectName("label_140")
        self.horizontalLayout_110.addWidget(self.label_140)
        self.formLayout_38.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_110
        )
        self.horizontalLayout_111 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_111.setSpacing(8)
        self.horizontalLayout_111.setObjectName("horizontalLayout_111")
        self.cropBurnYes_6 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_6.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_6.setSizePolicy(sizePolicy)
        self.cropBurnYes_6.setObjectName("cropBurnYes_6")
        self.horizontalLayout_111.addWidget(self.cropBurnYes_6)
        self.cropBurnNo_6 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_6.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_6.setSizePolicy(sizePolicy)
        self.cropBurnNo_6.setObjectName("cropBurnNo_6")
        self.horizontalLayout_111.addWidget(self.cropBurnNo_6)
        self.formLayout_38.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_111
        )
        self.label_9 = QtWidgets.QLabel(self.cropMgmtWidget_base_7)
        self.label_9.setObjectName("label_9")
        self.formLayout_38.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.gridLayout_31.addLayout(self.formLayout_38, 1, 0, 1, 1)
        self.verticalLayout_73.addWidget(self.cropMgmtWidget_base_7)
        self.stackedWidget.addWidget(self.cropMgmtPage_6)
        self.cropMgmtPage_7 = QtWidgets.QWidget()
        self.cropMgmtPage_7.setObjectName("cropMgmtPage_7")
        self.verticalLayout_75 = QtWidgets.QVBoxLayout(self.cropMgmtPage_7)
        self.verticalLayout_75.setObjectName("verticalLayout_75")
        self.cropMgmtWidget_base_8 = QtWidgets.QWidget(self.cropMgmtPage_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_8.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_8.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_8.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_8.setObjectName("cropMgmtWidget_base_8")
        self.gridLayout_32 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_8)
        self.gridLayout_32.setObjectName("gridLayout_32")
        self.label_141 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_141.sizePolicy().hasHeightForWidth())
        self.label_141.setSizePolicy(sizePolicy)
        self.label_141.setMinimumSize(QtCore.QSize(0, 30))
        self.label_141.setObjectName("label_141")
        self.gridLayout_32.addWidget(self.label_141, 0, 0, 1, 1)
        self.formLayout_39 = QtWidgets.QFormLayout()
        self.formLayout_39.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_39.setVerticalSpacing(15)
        self.formLayout_39.setObjectName("formLayout_39")
        spacerItem11 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_39.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem11)
        self.label_142 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.label_142.setObjectName("label_142")
        self.formLayout_39.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_142)
        self.cropType_7 = QtWidgets.QComboBox(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_7.sizePolicy().hasHeightForWidth())
        self.cropType_7.setSizePolicy(sizePolicy)
        self.cropType_7.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_7.setObjectName("cropType_7")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.setItemText(1, "")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.setItemText(19, "")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.cropType_7.setItemText(29, "")
        self.cropType_7.addItem("")
        self.cropType_7.addItem("")
        self.formLayout_39.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_7
        )
        self.cropInputType_7 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.cropInputType_7.setEnabled(False)
        self.cropInputType_7.setObjectName("cropInputType_7")
        self.formLayout_39.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_7
        )
        self.verticalLayout_74 = QtWidgets.QVBoxLayout()
        self.verticalLayout_74.setSpacing(8)
        self.verticalLayout_74.setObjectName("verticalLayout_74")
        self.cropCsvButton_7 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_8)
        self.cropCsvButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_7.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_7.setSizePolicy(sizePolicy)
        self.cropCsvButton_7.setObjectName("cropCsvButton_7")
        self.verticalLayout_74.addWidget(self.cropCsvButton_7)
        self.cropCustomButton_7 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_8)
        self.cropCustomButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_7.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_7.setSizePolicy(sizePolicy)
        self.cropCustomButton_7.setObjectName("cropCustomButton_7")
        self.verticalLayout_74.addWidget(self.cropCustomButton_7)
        self.formLayout_39.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_74
        )
        self.cropCsvLabel_7 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.cropCsvLabel_7.setEnabled(False)
        self.cropCsvLabel_7.setObjectName("cropCsvLabel_7")
        self.formLayout_39.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_7
        )
        self.horizontalLayout_112 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_112.setSpacing(8)
        self.horizontalLayout_112.setObjectName("horizontalLayout_112")
        self.cropCsvBrowseBox_7 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_8)
        self.cropCsvBrowseBox_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_7.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_7.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_7.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_7.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_7.setMaxLength(1000)
        self.cropCsvBrowseBox_7.setObjectName("cropCsvBrowseBox_7")
        self.horizontalLayout_112.addWidget(self.cropCsvBrowseBox_7)
        self.cropCsvBrowseButton_7 = QtWidgets.QPushButton(self.cropMgmtWidget_base_8)
        self.cropCsvBrowseButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_7.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_7.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_7.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_7.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_7.setObjectName("cropCsvBrowseButton_7")
        self.horizontalLayout_112.addWidget(self.cropCsvBrowseButton_7)
        self.formLayout_39.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_112
        )
        self.label_143 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.label_143.setEnabled(True)
        self.label_143.setObjectName("label_143")
        self.formLayout_39.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_143)
        self.horizontalLayout_113 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_113.setSpacing(8)
        self.horizontalLayout_113.setObjectName("horizontalLayout_113")
        self.cropYield_7 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_7.sizePolicy().hasHeightForWidth())
        self.cropYield_7.setSizePolicy(sizePolicy)
        self.cropYield_7.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_7.setDecimals(1)
        self.cropYield_7.setMaximum(20.0)
        self.cropYield_7.setSingleStep(0.5)
        self.cropYield_7.setProperty("value", 1.0)
        self.cropYield_7.setObjectName("cropYield_7")
        self.horizontalLayout_113.addWidget(self.cropYield_7)
        self.label_144 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_144.sizePolicy().hasHeightForWidth())
        self.label_144.setSizePolicy(sizePolicy)
        self.label_144.setObjectName("label_144")
        self.horizontalLayout_113.addWidget(self.label_144)
        self.formLayout_39.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_113
        )
        self.label_145 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.label_145.setObjectName("label_145")
        self.formLayout_39.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_145)
        self.horizontalLayout_114 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_114.setSpacing(8)
        self.horizontalLayout_114.setObjectName("horizontalLayout_114")
        self.leftInField_7 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_7.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_7.setSizePolicy(sizePolicy)
        self.leftInField_7.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_7.setMaximum(100)
        self.leftInField_7.setSingleStep(10)
        self.leftInField_7.setProperty("value", 100)
        self.leftInField_7.setObjectName("leftInField_7")
        self.horizontalLayout_114.addWidget(self.leftInField_7)
        self.label_146 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_146.sizePolicy().hasHeightForWidth())
        self.label_146.setSizePolicy(sizePolicy)
        self.label_146.setObjectName("label_146")
        self.horizontalLayout_114.addWidget(self.label_146)
        self.formLayout_39.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_114
        )
        self.horizontalLayout_115 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_115.setSpacing(8)
        self.horizontalLayout_115.setObjectName("horizontalLayout_115")
        self.cropBurnYes_7 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_7.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_7.setSizePolicy(sizePolicy)
        self.cropBurnYes_7.setObjectName("cropBurnYes_7")
        self.horizontalLayout_115.addWidget(self.cropBurnYes_7)
        self.cropBurnNo_7 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_7.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_7.setSizePolicy(sizePolicy)
        self.cropBurnNo_7.setObjectName("cropBurnNo_7")
        self.horizontalLayout_115.addWidget(self.cropBurnNo_7)
        self.formLayout_39.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_115
        )
        self.label_12 = QtWidgets.QLabel(self.cropMgmtWidget_base_8)
        self.label_12.setObjectName("label_12")
        self.formLayout_39.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.gridLayout_32.addLayout(self.formLayout_39, 1, 0, 1, 1)
        self.verticalLayout_75.addWidget(self.cropMgmtWidget_base_8)
        self.stackedWidget.addWidget(self.cropMgmtPage_7)
        self.cropMgmtPage_8 = QtWidgets.QWidget()
        self.cropMgmtPage_8.setObjectName("cropMgmtPage_8")
        self.verticalLayout_77 = QtWidgets.QVBoxLayout(self.cropMgmtPage_8)
        self.verticalLayout_77.setObjectName("verticalLayout_77")
        self.cropMgmtWidget_base_9 = QtWidgets.QWidget(self.cropMgmtPage_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_9.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_9.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_9.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_9.setObjectName("cropMgmtWidget_base_9")
        self.gridLayout_33 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_9)
        self.gridLayout_33.setObjectName("gridLayout_33")
        self.label_147 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_147.sizePolicy().hasHeightForWidth())
        self.label_147.setSizePolicy(sizePolicy)
        self.label_147.setMinimumSize(QtCore.QSize(0, 30))
        self.label_147.setObjectName("label_147")
        self.gridLayout_33.addWidget(self.label_147, 0, 0, 1, 1)
        self.formLayout_40 = QtWidgets.QFormLayout()
        self.formLayout_40.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_40.setVerticalSpacing(15)
        self.formLayout_40.setObjectName("formLayout_40")
        spacerItem12 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_40.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem12)
        self.label_148 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.label_148.setObjectName("label_148")
        self.formLayout_40.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_148)
        self.cropType_8 = QtWidgets.QComboBox(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_8.sizePolicy().hasHeightForWidth())
        self.cropType_8.setSizePolicy(sizePolicy)
        self.cropType_8.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_8.setObjectName("cropType_8")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.setItemText(1, "")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.setItemText(19, "")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.cropType_8.setItemText(29, "")
        self.cropType_8.addItem("")
        self.cropType_8.addItem("")
        self.formLayout_40.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_8
        )
        self.cropInputType_8 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.cropInputType_8.setEnabled(False)
        self.cropInputType_8.setObjectName("cropInputType_8")
        self.formLayout_40.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_8
        )
        self.verticalLayout_76 = QtWidgets.QVBoxLayout()
        self.verticalLayout_76.setSpacing(8)
        self.verticalLayout_76.setObjectName("verticalLayout_76")
        self.cropCsvButton_8 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_9)
        self.cropCsvButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_8.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_8.setSizePolicy(sizePolicy)
        self.cropCsvButton_8.setObjectName("cropCsvButton_8")
        self.verticalLayout_76.addWidget(self.cropCsvButton_8)
        self.cropCustomButton_8 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_9)
        self.cropCustomButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_8.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_8.setSizePolicy(sizePolicy)
        self.cropCustomButton_8.setObjectName("cropCustomButton_8")
        self.verticalLayout_76.addWidget(self.cropCustomButton_8)
        self.formLayout_40.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_76
        )
        self.cropCsvLabel_8 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.cropCsvLabel_8.setEnabled(False)
        self.cropCsvLabel_8.setObjectName("cropCsvLabel_8")
        self.formLayout_40.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_8
        )
        self.horizontalLayout_116 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_116.setSpacing(8)
        self.horizontalLayout_116.setObjectName("horizontalLayout_116")
        self.cropCsvBrowseBox_8 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_9)
        self.cropCsvBrowseBox_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_8.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_8.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_8.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_8.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_8.setMaxLength(1000)
        self.cropCsvBrowseBox_8.setObjectName("cropCsvBrowseBox_8")
        self.horizontalLayout_116.addWidget(self.cropCsvBrowseBox_8)
        self.cropCsvBrowseButton_8 = QtWidgets.QPushButton(self.cropMgmtWidget_base_9)
        self.cropCsvBrowseButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_8.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_8.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_8.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_8.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_8.setObjectName("cropCsvBrowseButton_8")
        self.horizontalLayout_116.addWidget(self.cropCsvBrowseButton_8)
        self.formLayout_40.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_116
        )
        self.label_149 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.label_149.setEnabled(True)
        self.label_149.setObjectName("label_149")
        self.formLayout_40.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_149)
        self.horizontalLayout_117 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_117.setSpacing(8)
        self.horizontalLayout_117.setObjectName("horizontalLayout_117")
        self.cropYield_8 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_8.sizePolicy().hasHeightForWidth())
        self.cropYield_8.setSizePolicy(sizePolicy)
        self.cropYield_8.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_8.setDecimals(1)
        self.cropYield_8.setMaximum(20.0)
        self.cropYield_8.setSingleStep(0.5)
        self.cropYield_8.setProperty("value", 1.0)
        self.cropYield_8.setObjectName("cropYield_8")
        self.horizontalLayout_117.addWidget(self.cropYield_8)
        self.label_150 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_150.sizePolicy().hasHeightForWidth())
        self.label_150.setSizePolicy(sizePolicy)
        self.label_150.setObjectName("label_150")
        self.horizontalLayout_117.addWidget(self.label_150)
        self.formLayout_40.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_117
        )
        self.label_151 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.label_151.setObjectName("label_151")
        self.formLayout_40.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_151)
        self.horizontalLayout_118 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_118.setSpacing(8)
        self.horizontalLayout_118.setObjectName("horizontalLayout_118")
        self.leftInField_8 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_8.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_8.setSizePolicy(sizePolicy)
        self.leftInField_8.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_8.setMaximum(100)
        self.leftInField_8.setSingleStep(10)
        self.leftInField_8.setProperty("value", 100)
        self.leftInField_8.setObjectName("leftInField_8")
        self.horizontalLayout_118.addWidget(self.leftInField_8)
        self.label_152 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_152.sizePolicy().hasHeightForWidth())
        self.label_152.setSizePolicy(sizePolicy)
        self.label_152.setObjectName("label_152")
        self.horizontalLayout_118.addWidget(self.label_152)
        self.formLayout_40.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_118
        )
        self.horizontalLayout_119 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_119.setSpacing(8)
        self.horizontalLayout_119.setObjectName("horizontalLayout_119")
        self.cropBurnYes_8 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_8.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_8.setSizePolicy(sizePolicy)
        self.cropBurnYes_8.setObjectName("cropBurnYes_8")
        self.horizontalLayout_119.addWidget(self.cropBurnYes_8)
        self.cropBurnNo_8 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_8.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_8.setSizePolicy(sizePolicy)
        self.cropBurnNo_8.setObjectName("cropBurnNo_8")
        self.horizontalLayout_119.addWidget(self.cropBurnNo_8)
        self.formLayout_40.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_119
        )
        self.label_13 = QtWidgets.QLabel(self.cropMgmtWidget_base_9)
        self.label_13.setObjectName("label_13")
        self.formLayout_40.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.gridLayout_33.addLayout(self.formLayout_40, 1, 0, 1, 1)
        self.verticalLayout_77.addWidget(self.cropMgmtWidget_base_9)
        self.stackedWidget.addWidget(self.cropMgmtPage_8)
        self.cropMgmtPage_9 = QtWidgets.QWidget()
        self.cropMgmtPage_9.setObjectName("cropMgmtPage_9")
        self.verticalLayout_79 = QtWidgets.QVBoxLayout(self.cropMgmtPage_9)
        self.verticalLayout_79.setObjectName("verticalLayout_79")
        self.cropMgmtWidget_base_10 = QtWidgets.QWidget(self.cropMgmtPage_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropMgmtWidget_base_10.sizePolicy().hasHeightForWidth()
        )
        self.cropMgmtWidget_base_10.setSizePolicy(sizePolicy)
        self.cropMgmtWidget_base_10.setMinimumSize(QtCore.QSize(620, 300))
        self.cropMgmtWidget_base_10.setObjectName("cropMgmtWidget_base_10")
        self.gridLayout_34 = QtWidgets.QGridLayout(self.cropMgmtWidget_base_10)
        self.gridLayout_34.setObjectName("gridLayout_34")
        self.label_153 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_153.sizePolicy().hasHeightForWidth())
        self.label_153.setSizePolicy(sizePolicy)
        self.label_153.setMinimumSize(QtCore.QSize(0, 30))
        self.label_153.setObjectName("label_153")
        self.gridLayout_34.addWidget(self.label_153, 0, 0, 1, 1)
        self.formLayout_41 = QtWidgets.QFormLayout()
        self.formLayout_41.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_41.setVerticalSpacing(15)
        self.formLayout_41.setObjectName("formLayout_41")
        spacerItem13 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_41.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem13)
        self.label_154 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.label_154.setObjectName("label_154")
        self.formLayout_41.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_154)
        self.cropType_9 = QtWidgets.QComboBox(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropType_9.sizePolicy().hasHeightForWidth())
        self.cropType_9.setSizePolicy(sizePolicy)
        self.cropType_9.setMinimumSize(QtCore.QSize(200, 0))
        self.cropType_9.setObjectName("cropType_9")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.setItemText(1, "")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.setItemText(19, "")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.cropType_9.setItemText(29, "")
        self.cropType_9.addItem("")
        self.cropType_9.addItem("")
        self.formLayout_41.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.cropType_9
        )
        self.cropInputType_9 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.cropInputType_9.setEnabled(False)
        self.cropInputType_9.setObjectName("cropInputType_9")
        self.formLayout_41.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.cropInputType_9
        )
        self.verticalLayout_78 = QtWidgets.QVBoxLayout()
        self.verticalLayout_78.setSpacing(8)
        self.verticalLayout_78.setObjectName("verticalLayout_78")
        self.cropCsvButton_9 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_10)
        self.cropCsvButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvButton_9.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvButton_9.setSizePolicy(sizePolicy)
        self.cropCsvButton_9.setObjectName("cropCsvButton_9")
        self.verticalLayout_78.addWidget(self.cropCsvButton_9)
        self.cropCustomButton_9 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_10)
        self.cropCustomButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCustomButton_9.sizePolicy().hasHeightForWidth()
        )
        self.cropCustomButton_9.setSizePolicy(sizePolicy)
        self.cropCustomButton_9.setObjectName("cropCustomButton_9")
        self.verticalLayout_78.addWidget(self.cropCustomButton_9)
        self.formLayout_41.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_78
        )
        self.cropCsvLabel_9 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.cropCsvLabel_9.setEnabled(False)
        self.cropCsvLabel_9.setObjectName("cropCsvLabel_9")
        self.formLayout_41.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.cropCsvLabel_9
        )
        self.horizontalLayout_120 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_120.setSpacing(8)
        self.horizontalLayout_120.setObjectName("horizontalLayout_120")
        self.cropCsvBrowseBox_9 = QtWidgets.QLineEdit(self.cropMgmtWidget_base_10)
        self.cropCsvBrowseBox_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseBox_9.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseBox_9.setSizePolicy(sizePolicy)
        self.cropCsvBrowseBox_9.setMinimumSize(QtCore.QSize(230, 0))
        self.cropCsvBrowseBox_9.setMaximumSize(QtCore.QSize(230, 16777215))
        self.cropCsvBrowseBox_9.setMaxLength(1000)
        self.cropCsvBrowseBox_9.setObjectName("cropCsvBrowseBox_9")
        self.horizontalLayout_120.addWidget(self.cropCsvBrowseBox_9)
        self.cropCsvBrowseButton_9 = QtWidgets.QPushButton(self.cropMgmtWidget_base_10)
        self.cropCsvBrowseButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropCsvBrowseButton_9.sizePolicy().hasHeightForWidth()
        )
        self.cropCsvBrowseButton_9.setSizePolicy(sizePolicy)
        self.cropCsvBrowseButton_9.setMinimumSize(QtCore.QSize(85, 0))
        self.cropCsvBrowseButton_9.setMaximumSize(QtCore.QSize(85, 16777215))
        self.cropCsvBrowseButton_9.setObjectName("cropCsvBrowseButton_9")
        self.horizontalLayout_120.addWidget(self.cropCsvBrowseButton_9)
        self.formLayout_41.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_120
        )
        self.label_155 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.label_155.setEnabled(True)
        self.label_155.setObjectName("label_155")
        self.formLayout_41.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_155)
        self.horizontalLayout_121 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_121.setSpacing(8)
        self.horizontalLayout_121.setObjectName("horizontalLayout_121")
        self.cropYield_9 = QtWidgets.QDoubleSpinBox(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropYield_9.sizePolicy().hasHeightForWidth())
        self.cropYield_9.setSizePolicy(sizePolicy)
        self.cropYield_9.setMinimumSize(QtCore.QSize(80, 0))
        self.cropYield_9.setDecimals(1)
        self.cropYield_9.setMaximum(20.0)
        self.cropYield_9.setSingleStep(0.5)
        self.cropYield_9.setProperty("value", 1.0)
        self.cropYield_9.setObjectName("cropYield_9")
        self.horizontalLayout_121.addWidget(self.cropYield_9)
        self.label_156 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_156.sizePolicy().hasHeightForWidth())
        self.label_156.setSizePolicy(sizePolicy)
        self.label_156.setObjectName("label_156")
        self.horizontalLayout_121.addWidget(self.label_156)
        self.formLayout_41.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_121
        )
        self.label_157 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.label_157.setObjectName("label_157")
        self.formLayout_41.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_157)
        self.horizontalLayout_122 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_122.setSpacing(8)
        self.horizontalLayout_122.setObjectName("horizontalLayout_122")
        self.leftInField_9 = QtWidgets.QSpinBox(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.leftInField_9.sizePolicy().hasHeightForWidth()
        )
        self.leftInField_9.setSizePolicy(sizePolicy)
        self.leftInField_9.setMinimumSize(QtCore.QSize(60, 0))
        self.leftInField_9.setMaximum(100)
        self.leftInField_9.setSingleStep(10)
        self.leftInField_9.setProperty("value", 100)
        self.leftInField_9.setObjectName("leftInField_9")
        self.horizontalLayout_122.addWidget(self.leftInField_9)
        self.label_158 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_158.sizePolicy().hasHeightForWidth())
        self.label_158.setSizePolicy(sizePolicy)
        self.label_158.setObjectName("label_158")
        self.horizontalLayout_122.addWidget(self.label_158)
        self.formLayout_41.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_122
        )
        self.horizontalLayout_123 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_123.setSpacing(8)
        self.horizontalLayout_123.setObjectName("horizontalLayout_123")
        self.cropBurnYes_9 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cropBurnYes_9.sizePolicy().hasHeightForWidth()
        )
        self.cropBurnYes_9.setSizePolicy(sizePolicy)
        self.cropBurnYes_9.setObjectName("cropBurnYes_9")
        self.horizontalLayout_123.addWidget(self.cropBurnYes_9)
        self.cropBurnNo_9 = QtWidgets.QRadioButton(self.cropMgmtWidget_base_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cropBurnNo_9.sizePolicy().hasHeightForWidth())
        self.cropBurnNo_9.setSizePolicy(sizePolicy)
        self.cropBurnNo_9.setObjectName("cropBurnNo_9")
        self.horizontalLayout_123.addWidget(self.cropBurnNo_9)
        self.formLayout_41.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_123
        )
        self.label_14 = QtWidgets.QLabel(self.cropMgmtWidget_base_10)
        self.label_14.setObjectName("label_14")
        self.formLayout_41.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.gridLayout_34.addLayout(self.formLayout_41, 1, 0, 1, 1)
        self.verticalLayout_79.addWidget(self.cropMgmtWidget_base_10)
        self.stackedWidget.addWidget(self.cropMgmtPage_9)
        self.treeMgmtPage_0 = QtWidgets.QWidget()
        self.treeMgmtPage_0.setObjectName("treeMgmtPage_0")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.treeMgmtPage_0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.widget_6 = QtWidgets.QWidget(self.treeMgmtPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_6.sizePolicy().hasHeightForWidth())
        self.widget_6.setSizePolicy(sizePolicy)
        self.widget_6.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_6.setObjectName("widget_6")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_50 = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_50.sizePolicy().hasHeightForWidth())
        self.label_50.setSizePolicy(sizePolicy)
        self.label_50.setMinimumSize(QtCore.QSize(0, 30))
        self.label_50.setObjectName("label_50")
        self.gridLayout_2.addWidget(self.label_50, 0, 0, 1, 1)
        self.formLayout_9 = QtWidgets.QFormLayout()
        self.formLayout_9.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_9.setVerticalSpacing(15)
        self.formLayout_9.setObjectName("formLayout_9")
        spacerItem14 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_9.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem14)
        self.label_6 = QtWidgets.QLabel(self.widget_6)
        self.label_6.setObjectName("label_6")
        self.formLayout_9.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.speciesName_0 = QtWidgets.QLineEdit(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(27)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_0.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_0.setSizePolicy(sizePolicy)
        self.speciesName_0.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_0.setObjectName("speciesName_0")
        self.formLayout_9.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_0
        )
        self.label_51 = QtWidgets.QLabel(self.widget_6)
        self.label_51.setObjectName("label_51")
        self.formLayout_9.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_51)
        self.treeType_0 = QtWidgets.QComboBox(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_0.sizePolicy().hasHeightForWidth())
        self.treeType_0.setSizePolicy(sizePolicy)
        self.treeType_0.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_0.setObjectName("treeType_0")
        self.treeType_0.addItem("")
        self.treeType_0.addItem("")
        self.treeType_0.setItemText(1, "")
        self.treeType_0.addItem("")
        self.treeType_0.addItem("")
        self.treeType_0.addItem("")
        self.treeType_0.setItemText(4, "")
        self.treeType_0.addItem("")
        self.treeType_0.addItem("")
        self.formLayout_9.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.treeType_0)
        self.treeInputType_0 = QtWidgets.QLabel(self.widget_6)
        self.treeInputType_0.setEnabled(False)
        self.treeInputType_0.setObjectName("treeInputType_0")
        self.formLayout_9.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_0
        )
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setSpacing(8)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.treeCsvButton_0 = QtWidgets.QRadioButton(self.widget_6)
        self.treeCsvButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_0.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_0.setSizePolicy(sizePolicy)
        self.treeCsvButton_0.setObjectName("treeCsvButton_0")
        self.verticalLayout_20.addWidget(self.treeCsvButton_0)
        self.treeCustomButton_0 = QtWidgets.QRadioButton(self.widget_6)
        self.treeCustomButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_0.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_0.setSizePolicy(sizePolicy)
        self.treeCustomButton_0.setObjectName("treeCustomButton_0")
        self.verticalLayout_20.addWidget(self.treeCustomButton_0)
        self.formLayout_9.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_20
        )
        self.treeCsvLabel_0 = QtWidgets.QLabel(self.widget_6)
        self.treeCsvLabel_0.setEnabled(False)
        self.treeCsvLabel_0.setObjectName("treeCsvLabel_0")
        self.formLayout_9.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_0
        )
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setSpacing(8)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.treeCsvBrowseBox_0 = QtWidgets.QLineEdit(self.widget_6)
        self.treeCsvBrowseBox_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_0.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_0.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_0.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_0.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_0.setMaxLength(1000)
        self.treeCsvBrowseBox_0.setObjectName("treeCsvBrowseBox_0")
        self.horizontalLayout_15.addWidget(self.treeCsvBrowseBox_0)
        self.treeCsvBrowseButton_0 = QtWidgets.QPushButton(self.widget_6)
        self.treeCsvBrowseButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_0.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_0.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_0.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_0.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_0.setObjectName("treeCsvBrowseButton_0")
        self.horizontalLayout_15.addWidget(self.treeCsvBrowseButton_0)
        self.formLayout_9.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_15
        )
        self.label_53 = QtWidgets.QLabel(self.widget_6)
        self.label_53.setObjectName("label_53")
        self.formLayout_9.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_53)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setSpacing(8)
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.standDens_0 = QtWidgets.QSpinBox(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_0.sizePolicy().hasHeightForWidth())
        self.standDens_0.setSizePolicy(sizePolicy)
        self.standDens_0.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_0.setMaximum(999)
        self.standDens_0.setSingleStep(50)
        self.standDens_0.setProperty("value", 200)
        self.standDens_0.setObjectName("standDens_0")
        self.horizontalLayout_24.addWidget(self.standDens_0)
        self.label_55 = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_55.sizePolicy().hasHeightForWidth())
        self.label_55.setSizePolicy(sizePolicy)
        self.label_55.setObjectName("label_55")
        self.horizontalLayout_24.addWidget(self.label_55)
        self.formLayout_9.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_24
        )
        self.label_17 = QtWidgets.QLabel(self.widget_6)
        self.label_17.setObjectName("label_17")
        self.formLayout_9.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setSpacing(8)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.label_20 = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_20.addWidget(self.label_20)
        self.yearPlanted_0 = QtWidgets.QSpinBox(self.widget_6)
        self.yearPlanted_0.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_0.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_0.setObjectName("yearPlanted_0")
        self.horizontalLayout_20.addWidget(self.yearPlanted_0)
        self.formLayout_9.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_20
        )
        self.gridLayout_2.addLayout(self.formLayout_9, 1, 0, 1, 1)
        self.verticalLayout_9.addWidget(self.widget_6)
        self.stackedWidget.addWidget(self.treeMgmtPage_0)
        self.treeThinPage_0 = QtWidgets.QWidget()
        self.treeThinPage_0.setObjectName("treeThinPage_0")
        self.verticalLayout_36 = QtWidgets.QVBoxLayout(self.treeThinPage_0)
        self.verticalLayout_36.setObjectName("verticalLayout_36")
        self.widget_8 = QtWidgets.QWidget(self.treeThinPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_8.sizePolicy().hasHeightForWidth())
        self.widget_8.setSizePolicy(sizePolicy)
        self.widget_8.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_8.setObjectName("widget_8")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.widget_8)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_54 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_54.sizePolicy().hasHeightForWidth())
        self.label_54.setSizePolicy(sizePolicy)
        self.label_54.setMinimumSize(QtCore.QSize(0, 30))
        self.label_54.setObjectName("label_54")
        self.gridLayout_7.addWidget(self.label_54, 0, 0, 1, 1)
        self.formLayout_12 = QtWidgets.QFormLayout()
        self.formLayout_12.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_12.setVerticalSpacing(15)
        self.formLayout_12.setObjectName("formLayout_12")
        spacerItem15 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_12.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem15)
        self.label_26 = QtWidgets.QLabel(self.widget_8)
        self.label_26.setObjectName("label_26")
        self.formLayout_12.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.verticalLayout_27 = QtWidgets.QVBoxLayout()
        self.verticalLayout_27.setObjectName("verticalLayout_27")
        self.thinNever_0 = QtWidgets.QRadioButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_0.sizePolicy().hasHeightForWidth())
        self.thinNever_0.setSizePolicy(sizePolicy)
        self.thinNever_0.setObjectName("thinNever_0")
        self.verticalLayout_27.addWidget(self.thinNever_0)
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setSpacing(8)
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.thinInterval_0 = QtWidgets.QRadioButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_0.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_0.setSizePolicy(sizePolicy)
        self.thinInterval_0.setObjectName("thinInterval_0")
        self.horizontalLayout_29.addWidget(self.thinInterval_0)
        self.thinFreq_0 = QtWidgets.QSpinBox(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_0.sizePolicy().hasHeightForWidth())
        self.thinFreq_0.setSizePolicy(sizePolicy)
        self.thinFreq_0.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_0.setMinimum(1)
        self.thinFreq_0.setSingleStep(10)
        self.thinFreq_0.setProperty("value", 1)
        self.thinFreq_0.setObjectName("thinFreq_0")
        self.horizontalLayout_29.addWidget(self.thinFreq_0)
        self.thinFreqLabel_0 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_0.setSizePolicy(sizePolicy)
        self.thinFreqLabel_0.setObjectName("thinFreqLabel_0")
        self.horizontalLayout_29.addWidget(self.thinFreqLabel_0)
        self.verticalLayout_27.addLayout(self.horizontalLayout_29)
        self.thinCustom_0 = QtWidgets.QRadioButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_0.sizePolicy().hasHeightForWidth())
        self.thinCustom_0.setSizePolicy(sizePolicy)
        self.thinCustom_0.setObjectName("thinCustom_0")
        self.verticalLayout_27.addWidget(self.thinCustom_0)
        self.formLayout_12.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_27
        )
        self.thinCustomText_0 = QtWidgets.QLabel(self.widget_8)
        self.thinCustomText_0.setObjectName("thinCustomText_0")
        self.formLayout_12.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_0
        )
        self.thinCustomLine_0 = QtWidgets.QLineEdit(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_0.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_0.setSizePolicy(sizePolicy)
        self.thinCustomLine_0.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_0.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_0.setObjectName("thinCustomLine_0")
        self.formLayout_12.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_0
        )
        self.treesThinnedText_0 = QtWidgets.QLabel(self.widget_8)
        self.treesThinnedText_0.setObjectName("treesThinnedText_0")
        self.formLayout_12.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_0
        )
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setSpacing(8)
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.treesThinned_0 = QtWidgets.QSpinBox(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_0.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_0.setSizePolicy(sizePolicy)
        self.treesThinned_0.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_0.setMaximum(100)
        self.treesThinned_0.setSingleStep(10)
        self.treesThinned_0.setProperty("value", 50)
        self.treesThinned_0.setObjectName("treesThinned_0")
        self.horizontalLayout_28.addWidget(self.treesThinned_0)
        self.treesThinnedLabel_0 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_0.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_0.setObjectName("treesThinnedLabel_0")
        self.horizontalLayout_28.addWidget(self.treesThinnedLabel_0)
        self.formLayout_12.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_28
        )
        self.thinnedLeftText_0 = QtWidgets.QLabel(self.widget_8)
        self.thinnedLeftText_0.setObjectName("thinnedLeftText_0")
        self.formLayout_12.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_0
        )
        self.thinnedStemsLeftText_0 = QtWidgets.QLabel(self.widget_8)
        self.thinnedStemsLeftText_0.setObjectName("thinnedStemsLeftText_0")
        self.formLayout_12.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_0
        )
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setSpacing(8)
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.thinnedStemsLeft_0 = QtWidgets.QSpinBox(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_0.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_0.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_0.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_0.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_0.setMaximum(100)
        self.thinnedStemsLeft_0.setSingleStep(10)
        self.thinnedStemsLeft_0.setObjectName("thinnedStemsLeft_0")
        self.horizontalLayout_30.addWidget(self.thinnedStemsLeft_0)
        self.thinnedStemsLeftLabel_0 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_0.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_0.setObjectName("thinnedStemsLeftLabel_0")
        self.horizontalLayout_30.addWidget(self.thinnedStemsLeftLabel_0)
        self.formLayout_12.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_30
        )
        self.thinnedBranchesLeftText_0 = QtWidgets.QLabel(self.widget_8)
        self.thinnedBranchesLeftText_0.setObjectName("thinnedBranchesLeftText_0")
        self.formLayout_12.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_0
        )
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_31.setSpacing(8)
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.thinnedBranchesLeft_0 = QtWidgets.QSpinBox(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_0.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_0.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_0.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_0.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_0.setMaximum(100)
        self.thinnedBranchesLeft_0.setSingleStep(10)
        self.thinnedBranchesLeft_0.setObjectName("thinnedBranchesLeft_0")
        self.horizontalLayout_31.addWidget(self.thinnedBranchesLeft_0)
        self.thinnedBranchesLeftLabel_0 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_0.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_0.setObjectName("thinnedBranchesLeftLabel_0")
        self.horizontalLayout_31.addWidget(self.thinnedBranchesLeftLabel_0)
        self.formLayout_12.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_31
        )
        self.gridLayout_7.addLayout(self.formLayout_12, 1, 0, 1, 1)
        self.verticalLayout_36.addWidget(self.widget_8)
        self.stackedWidget.addWidget(self.treeThinPage_0)
        self.treeMortPage_0 = QtWidgets.QWidget()
        self.treeMortPage_0.setObjectName("treeMortPage_0")
        self.verticalLayout_29 = QtWidgets.QVBoxLayout(self.treeMortPage_0)
        self.verticalLayout_29.setObjectName("verticalLayout_29")
        self.widget_9 = QtWidgets.QWidget(self.treeMortPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_9.sizePolicy().hasHeightForWidth())
        self.widget_9.setSizePolicy(sizePolicy)
        self.widget_9.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_9.setObjectName("widget_9")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.widget_9)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_58 = QtWidgets.QLabel(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_58.sizePolicy().hasHeightForWidth())
        self.label_58.setSizePolicy(sizePolicy)
        self.label_58.setMinimumSize(QtCore.QSize(0, 30))
        self.label_58.setObjectName("label_58")
        self.gridLayout_8.addWidget(self.label_58, 0, 0, 1, 1)
        self.formLayout_13 = QtWidgets.QFormLayout()
        self.formLayout_13.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_13.setVerticalSpacing(15)
        self.formLayout_13.setObjectName("formLayout_13")
        spacerItem16 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_13.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem16)
        self.label_66 = QtWidgets.QLabel(self.widget_9)
        self.label_66.setObjectName("label_66")
        self.formLayout_13.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_66)
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_32.setSpacing(8)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.treesDead_0 = QtWidgets.QSpinBox(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_0.sizePolicy().hasHeightForWidth())
        self.treesDead_0.setSizePolicy(sizePolicy)
        self.treesDead_0.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_0.setMaximum(100)
        self.treesDead_0.setSingleStep(10)
        self.treesDead_0.setProperty("value", 1)
        self.treesDead_0.setObjectName("treesDead_0")
        self.horizontalLayout_32.addWidget(self.treesDead_0)
        self.label_67 = QtWidgets.QLabel(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_67.sizePolicy().hasHeightForWidth())
        self.label_67.setSizePolicy(sizePolicy)
        self.label_67.setObjectName("label_67")
        self.horizontalLayout_32.addWidget(self.label_67)
        self.formLayout_13.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_32
        )
        self.deadLeftText_0 = QtWidgets.QLabel(self.widget_9)
        self.deadLeftText_0.setObjectName("deadLeftText_0")
        self.formLayout_13.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_0
        )
        self.deadStemsLeftText_0 = QtWidgets.QLabel(self.widget_9)
        self.deadStemsLeftText_0.setObjectName("deadStemsLeftText_0")
        self.formLayout_13.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_0
        )
        self.horizontalLayout_33 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_33.setSpacing(8)
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.deadStemsLeft_0 = QtWidgets.QSpinBox(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_0.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_0.setSizePolicy(sizePolicy)
        self.deadStemsLeft_0.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_0.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_0.setMaximum(100)
        self.deadStemsLeft_0.setSingleStep(10)
        self.deadStemsLeft_0.setObjectName("deadStemsLeft_0")
        self.horizontalLayout_33.addWidget(self.deadStemsLeft_0)
        self.deadStemsLeftLabel_0 = QtWidgets.QLabel(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_0.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_0.setObjectName("deadStemsLeftLabel_0")
        self.horizontalLayout_33.addWidget(self.deadStemsLeftLabel_0)
        self.formLayout_13.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_33
        )
        self.deadBranchesLeftText_0 = QtWidgets.QLabel(self.widget_9)
        self.deadBranchesLeftText_0.setObjectName("deadBranchesLeftText_0")
        self.formLayout_13.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_0
        )
        self.horizontalLayout_34 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_34.setSpacing(8)
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        self.deadBranchesLeft_0 = QtWidgets.QSpinBox(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_0.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_0.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_0.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_0.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_0.setMaximum(100)
        self.deadBranchesLeft_0.setSingleStep(10)
        self.deadBranchesLeft_0.setObjectName("deadBranchesLeft_0")
        self.horizontalLayout_34.addWidget(self.deadBranchesLeft_0)
        self.deadBranchesLeftLabel_0 = QtWidgets.QLabel(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_0.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_0.setObjectName("deadBranchesLeftLabel_0")
        self.horizontalLayout_34.addWidget(self.deadBranchesLeftLabel_0)
        self.formLayout_13.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_34
        )
        self.gridLayout_8.addLayout(self.formLayout_13, 1, 0, 1, 1)
        self.verticalLayout_29.addWidget(self.widget_9)
        self.stackedWidget.addWidget(self.treeMortPage_0)
        self.treeGrowthPage_0 = QtWidgets.QWidget()
        self.treeGrowthPage_0.setObjectName("treeGrowthPage_0")
        self.verticalLayout_37 = QtWidgets.QVBoxLayout(self.treeGrowthPage_0)
        self.verticalLayout_37.setObjectName("verticalLayout_37")
        self.widget_7 = QtWidgets.QWidget(self.treeGrowthPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy)
        self.widget_7.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_7.setObjectName("widget_7")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.widget_7)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.formLayout_10 = QtWidgets.QFormLayout()
        self.formLayout_10.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_10.setVerticalSpacing(15)
        self.formLayout_10.setObjectName("formLayout_10")
        spacerItem17 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_10.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem17)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setSpacing(8)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.growthCsvButton_0 = QtWidgets.QRadioButton(self.widget_7)
        self.growthCsvButton_0.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_0.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_0.setSizePolicy(sizePolicy)
        self.growthCsvButton_0.setObjectName("growthCsvButton_0")
        self.verticalLayout_10.addWidget(self.growthCsvButton_0)
        self.growthCustomButton_0 = QtWidgets.QRadioButton(self.widget_7)
        self.growthCustomButton_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_0.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_0.setSizePolicy(sizePolicy)
        self.growthCustomButton_0.setObjectName("growthCustomButton_0")
        self.verticalLayout_10.addWidget(self.growthCustomButton_0)
        self.formLayout_10.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_10
        )
        self.growthCsvLabel_0 = QtWidgets.QLabel(self.widget_7)
        self.growthCsvLabel_0.setEnabled(True)
        self.growthCsvLabel_0.setObjectName("growthCsvLabel_0")
        self.formLayout_10.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_0
        )
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setSpacing(8)
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.growthCsvBrowseBox_0 = QtWidgets.QLineEdit(self.widget_7)
        self.growthCsvBrowseBox_0.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_0.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_0.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_0.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_0.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_0.setMaxLength(1000)
        self.growthCsvBrowseBox_0.setObjectName("growthCsvBrowseBox_0")
        self.horizontalLayout_25.addWidget(self.growthCsvBrowseBox_0)
        self.growthCsvBrowseButton_0 = QtWidgets.QPushButton(self.widget_7)
        self.growthCsvBrowseButton_0.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_0.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_0.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_0.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_0.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_0.setObjectName("growthCsvBrowseButton_0")
        self.horizontalLayout_25.addWidget(self.growthCsvBrowseButton_0)
        self.formLayout_10.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_25
        )
        self.label_52 = QtWidgets.QLabel(self.widget_7)
        self.label_52.setObjectName("label_52")
        self.formLayout_10.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_52)
        self.allom_0 = QtWidgets.QComboBox(self.widget_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_0.sizePolicy().hasHeightForWidth())
        self.allom_0.setSizePolicy(sizePolicy)
        self.allom_0.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_0.setObjectName("allom_0")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.setItemText(1, "")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.setItemText(7, "")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.allom_0.setItemText(12, "")
        self.allom_0.addItem("")
        self.allom_0.addItem("")
        self.formLayout_10.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_0)
        self.gridLayout_18.addLayout(self.formLayout_10, 1, 0, 1, 1)
        self.label_56 = QtWidgets.QLabel(self.widget_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_56.sizePolicy().hasHeightForWidth())
        self.label_56.setSizePolicy(sizePolicy)
        self.label_56.setMinimumSize(QtCore.QSize(0, 30))
        self.label_56.setObjectName("label_56")
        self.gridLayout_18.addWidget(self.label_56, 0, 0, 1, 1)
        self.verticalLayout_37.addWidget(self.widget_7)
        self.stackedWidget.addWidget(self.treeGrowthPage_0)
        self.treeMgmtPage_1 = QtWidgets.QWidget()
        self.treeMgmtPage_1.setObjectName("treeMgmtPage_1")
        self.verticalLayout_42 = QtWidgets.QVBoxLayout(self.treeMgmtPage_1)
        self.verticalLayout_42.setObjectName("verticalLayout_42")
        self.widget_14 = QtWidgets.QWidget(self.treeMgmtPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_14.sizePolicy().hasHeightForWidth())
        self.widget_14.setSizePolicy(sizePolicy)
        self.widget_14.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_14.setObjectName("widget_14")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_14)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_89 = QtWidgets.QLabel(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_89.sizePolicy().hasHeightForWidth())
        self.label_89.setSizePolicy(sizePolicy)
        self.label_89.setMinimumSize(QtCore.QSize(0, 30))
        self.label_89.setObjectName("label_89")
        self.gridLayout_4.addWidget(self.label_89, 0, 0, 1, 1)
        self.formLayout_21 = QtWidgets.QFormLayout()
        self.formLayout_21.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_21.setVerticalSpacing(15)
        self.formLayout_21.setObjectName("formLayout_21")
        spacerItem18 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_21.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem18)
        self.label_90 = QtWidgets.QLabel(self.widget_14)
        self.label_90.setObjectName("label_90")
        self.formLayout_21.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_90)
        self.treeType_1 = QtWidgets.QComboBox(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_1.sizePolicy().hasHeightForWidth())
        self.treeType_1.setSizePolicy(sizePolicy)
        self.treeType_1.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_1.setObjectName("treeType_1")
        self.treeType_1.addItem("")
        self.treeType_1.addItem("")
        self.treeType_1.setItemText(1, "")
        self.treeType_1.addItem("")
        self.treeType_1.addItem("")
        self.treeType_1.addItem("")
        self.treeType_1.setItemText(4, "")
        self.treeType_1.addItem("")
        self.treeType_1.addItem("")
        self.formLayout_21.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_1
        )
        self.treeInputType_1 = QtWidgets.QLabel(self.widget_14)
        self.treeInputType_1.setEnabled(False)
        self.treeInputType_1.setObjectName("treeInputType_1")
        self.formLayout_21.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_1
        )
        self.verticalLayout_41 = QtWidgets.QVBoxLayout()
        self.verticalLayout_41.setSpacing(8)
        self.verticalLayout_41.setObjectName("verticalLayout_41")
        self.treeCsvButton_1 = QtWidgets.QRadioButton(self.widget_14)
        self.treeCsvButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_1.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_1.setSizePolicy(sizePolicy)
        self.treeCsvButton_1.setObjectName("treeCsvButton_1")
        self.verticalLayout_41.addWidget(self.treeCsvButton_1)
        self.treeCustomButton_1 = QtWidgets.QRadioButton(self.widget_14)
        self.treeCustomButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_1.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_1.setSizePolicy(sizePolicy)
        self.treeCustomButton_1.setObjectName("treeCustomButton_1")
        self.verticalLayout_41.addWidget(self.treeCustomButton_1)
        self.formLayout_21.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_41
        )
        self.treeCsvLabel_1 = QtWidgets.QLabel(self.widget_14)
        self.treeCsvLabel_1.setEnabled(False)
        self.treeCsvLabel_1.setObjectName("treeCsvLabel_1")
        self.formLayout_21.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_1
        )
        self.horizontalLayout_60 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_60.setSpacing(8)
        self.horizontalLayout_60.setObjectName("horizontalLayout_60")
        self.treeCsvBrowseBox_1 = QtWidgets.QLineEdit(self.widget_14)
        self.treeCsvBrowseBox_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_1.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_1.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_1.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_1.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_1.setMaxLength(1000)
        self.treeCsvBrowseBox_1.setObjectName("treeCsvBrowseBox_1")
        self.horizontalLayout_60.addWidget(self.treeCsvBrowseBox_1)
        self.treeCsvBrowseButton_1 = QtWidgets.QPushButton(self.widget_14)
        self.treeCsvBrowseButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_1.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_1.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_1.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_1.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_1.setObjectName("treeCsvBrowseButton_1")
        self.horizontalLayout_60.addWidget(self.treeCsvBrowseButton_1)
        self.formLayout_21.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_60
        )
        self.label_91 = QtWidgets.QLabel(self.widget_14)
        self.label_91.setObjectName("label_91")
        self.formLayout_21.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_91)
        self.horizontalLayout_61 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_61.setSpacing(8)
        self.horizontalLayout_61.setObjectName("horizontalLayout_61")
        self.standDens_1 = QtWidgets.QSpinBox(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_1.sizePolicy().hasHeightForWidth())
        self.standDens_1.setSizePolicy(sizePolicy)
        self.standDens_1.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_1.setMaximum(999)
        self.standDens_1.setSingleStep(50)
        self.standDens_1.setProperty("value", 200)
        self.standDens_1.setObjectName("standDens_1")
        self.horizontalLayout_61.addWidget(self.standDens_1)
        self.label_92 = QtWidgets.QLabel(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_92.sizePolicy().hasHeightForWidth())
        self.label_92.setSizePolicy(sizePolicy)
        self.label_92.setObjectName("label_92")
        self.horizontalLayout_61.addWidget(self.label_92)
        self.formLayout_21.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_61
        )
        self.label_18 = QtWidgets.QLabel(self.widget_14)
        self.label_18.setObjectName("label_18")
        self.formLayout_21.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.horizontalLayout_62 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_62.setSpacing(8)
        self.horizontalLayout_62.setObjectName("horizontalLayout_62")
        self.label_21 = QtWidgets.QLabel(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_62.addWidget(self.label_21)
        self.yearPlanted_1 = QtWidgets.QSpinBox(self.widget_14)
        self.yearPlanted_1.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_1.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_1.setObjectName("yearPlanted_1")
        self.horizontalLayout_62.addWidget(self.yearPlanted_1)
        self.formLayout_21.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_62
        )
        self.label_226 = QtWidgets.QLabel(self.widget_14)
        self.label_226.setObjectName("label_226")
        self.formLayout_21.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_226)
        self.speciesName_1 = QtWidgets.QLineEdit(self.widget_14)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_1.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_1.setSizePolicy(sizePolicy)
        self.speciesName_1.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_1.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_1.setObjectName("speciesName_1")
        self.formLayout_21.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_1
        )
        self.gridLayout_4.addLayout(self.formLayout_21, 1, 0, 1, 1)
        self.verticalLayout_42.addWidget(self.widget_14)
        self.stackedWidget.addWidget(self.treeMgmtPage_1)
        self.treeThinPage_1 = QtWidgets.QWidget()
        self.treeThinPage_1.setObjectName("treeThinPage_1")
        self.verticalLayout_44 = QtWidgets.QVBoxLayout(self.treeThinPage_1)
        self.verticalLayout_44.setObjectName("verticalLayout_44")
        self.widget_15 = QtWidgets.QWidget(self.treeThinPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_15.sizePolicy().hasHeightForWidth())
        self.widget_15.setSizePolicy(sizePolicy)
        self.widget_15.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_15.setObjectName("widget_15")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.widget_15)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.label_93 = QtWidgets.QLabel(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_93.sizePolicy().hasHeightForWidth())
        self.label_93.setSizePolicy(sizePolicy)
        self.label_93.setMinimumSize(QtCore.QSize(0, 30))
        self.label_93.setObjectName("label_93")
        self.gridLayout_13.addWidget(self.label_93, 0, 0, 1, 1)
        self.formLayout_22 = QtWidgets.QFormLayout()
        self.formLayout_22.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_22.setVerticalSpacing(15)
        self.formLayout_22.setObjectName("formLayout_22")
        spacerItem19 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_22.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem19)
        self.label_32 = QtWidgets.QLabel(self.widget_15)
        self.label_32.setObjectName("label_32")
        self.formLayout_22.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_32)
        self.verticalLayout_43 = QtWidgets.QVBoxLayout()
        self.verticalLayout_43.setObjectName("verticalLayout_43")
        self.thinNever_1 = QtWidgets.QRadioButton(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_1.sizePolicy().hasHeightForWidth())
        self.thinNever_1.setSizePolicy(sizePolicy)
        self.thinNever_1.setObjectName("thinNever_1")
        self.verticalLayout_43.addWidget(self.thinNever_1)
        self.horizontalLayout_64 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_64.setSpacing(8)
        self.horizontalLayout_64.setObjectName("horizontalLayout_64")
        self.thinInterval_1 = QtWidgets.QRadioButton(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_1.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_1.setSizePolicy(sizePolicy)
        self.thinInterval_1.setObjectName("thinInterval_1")
        self.horizontalLayout_64.addWidget(self.thinInterval_1)
        self.thinFreq_1 = QtWidgets.QSpinBox(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_1.sizePolicy().hasHeightForWidth())
        self.thinFreq_1.setSizePolicy(sizePolicy)
        self.thinFreq_1.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_1.setMinimum(1)
        self.thinFreq_1.setSingleStep(10)
        self.thinFreq_1.setProperty("value", 1)
        self.thinFreq_1.setObjectName("thinFreq_1")
        self.horizontalLayout_64.addWidget(self.thinFreq_1)
        self.thinFreqLabel_1 = QtWidgets.QLabel(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_1.setSizePolicy(sizePolicy)
        self.thinFreqLabel_1.setObjectName("thinFreqLabel_1")
        self.horizontalLayout_64.addWidget(self.thinFreqLabel_1)
        self.verticalLayout_43.addLayout(self.horizontalLayout_64)
        self.thinCustom_1 = QtWidgets.QRadioButton(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_1.sizePolicy().hasHeightForWidth())
        self.thinCustom_1.setSizePolicy(sizePolicy)
        self.thinCustom_1.setObjectName("thinCustom_1")
        self.verticalLayout_43.addWidget(self.thinCustom_1)
        self.formLayout_22.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_43
        )
        self.thinCustomText_1 = QtWidgets.QLabel(self.widget_15)
        self.thinCustomText_1.setObjectName("thinCustomText_1")
        self.formLayout_22.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_1
        )
        self.thinCustomLine_1 = QtWidgets.QLineEdit(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_1.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_1.setSizePolicy(sizePolicy)
        self.thinCustomLine_1.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_1.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_1.setObjectName("thinCustomLine_1")
        self.formLayout_22.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_1
        )
        self.treesThinnedText_1 = QtWidgets.QLabel(self.widget_15)
        self.treesThinnedText_1.setObjectName("treesThinnedText_1")
        self.formLayout_22.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_1
        )
        self.horizontalLayout_65 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_65.setSpacing(8)
        self.horizontalLayout_65.setObjectName("horizontalLayout_65")
        self.treesThinned_1 = QtWidgets.QSpinBox(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_1.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_1.setSizePolicy(sizePolicy)
        self.treesThinned_1.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_1.setMaximum(100)
        self.treesThinned_1.setSingleStep(10)
        self.treesThinned_1.setProperty("value", 50)
        self.treesThinned_1.setObjectName("treesThinned_1")
        self.horizontalLayout_65.addWidget(self.treesThinned_1)
        self.treesThinnedLabel_1 = QtWidgets.QLabel(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_1.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_1.setObjectName("treesThinnedLabel_1")
        self.horizontalLayout_65.addWidget(self.treesThinnedLabel_1)
        self.formLayout_22.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_65
        )
        self.thinnedLeftText_1 = QtWidgets.QLabel(self.widget_15)
        self.thinnedLeftText_1.setObjectName("thinnedLeftText_1")
        self.formLayout_22.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_1
        )
        self.thinnedStemsLeftText_1 = QtWidgets.QLabel(self.widget_15)
        self.thinnedStemsLeftText_1.setObjectName("thinnedStemsLeftText_1")
        self.formLayout_22.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_1
        )
        self.horizontalLayout_66 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_66.setSpacing(8)
        self.horizontalLayout_66.setObjectName("horizontalLayout_66")
        self.thinnedStemsLeft_1 = QtWidgets.QSpinBox(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_1.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_1.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_1.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_1.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_1.setMaximum(100)
        self.thinnedStemsLeft_1.setSingleStep(10)
        self.thinnedStemsLeft_1.setObjectName("thinnedStemsLeft_1")
        self.horizontalLayout_66.addWidget(self.thinnedStemsLeft_1)
        self.thinnedStemsLeftLabel_1 = QtWidgets.QLabel(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_1.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_1.setObjectName("thinnedStemsLeftLabel_1")
        self.horizontalLayout_66.addWidget(self.thinnedStemsLeftLabel_1)
        self.formLayout_22.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_66
        )
        self.thinnedBranchesLeftText_1 = QtWidgets.QLabel(self.widget_15)
        self.thinnedBranchesLeftText_1.setObjectName("thinnedBranchesLeftText_1")
        self.formLayout_22.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_1
        )
        self.horizontalLayout_67 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_67.setSpacing(8)
        self.horizontalLayout_67.setObjectName("horizontalLayout_67")
        self.thinnedBranchesLeft_1 = QtWidgets.QSpinBox(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_1.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_1.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_1.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_1.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_1.setMaximum(100)
        self.thinnedBranchesLeft_1.setSingleStep(10)
        self.thinnedBranchesLeft_1.setObjectName("thinnedBranchesLeft_1")
        self.horizontalLayout_67.addWidget(self.thinnedBranchesLeft_1)
        self.thinnedBranchesLeftLabel_1 = QtWidgets.QLabel(self.widget_15)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_1.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_1.setObjectName("thinnedBranchesLeftLabel_1")
        self.horizontalLayout_67.addWidget(self.thinnedBranchesLeftLabel_1)
        self.formLayout_22.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_67
        )
        self.gridLayout_13.addLayout(self.formLayout_22, 1, 0, 1, 1)
        self.verticalLayout_44.addWidget(self.widget_15)
        self.stackedWidget.addWidget(self.treeThinPage_1)
        self.treeMortPage_1 = QtWidgets.QWidget()
        self.treeMortPage_1.setObjectName("treeMortPage_1")
        self.verticalLayout_45 = QtWidgets.QVBoxLayout(self.treeMortPage_1)
        self.verticalLayout_45.setObjectName("verticalLayout_45")
        self.widget_16 = QtWidgets.QWidget(self.treeMortPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_16.sizePolicy().hasHeightForWidth())
        self.widget_16.setSizePolicy(sizePolicy)
        self.widget_16.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_16.setObjectName("widget_16")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.widget_16)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.label_94 = QtWidgets.QLabel(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_94.sizePolicy().hasHeightForWidth())
        self.label_94.setSizePolicy(sizePolicy)
        self.label_94.setMinimumSize(QtCore.QSize(0, 30))
        self.label_94.setObjectName("label_94")
        self.gridLayout_14.addWidget(self.label_94, 0, 0, 1, 1)
        self.formLayout_23 = QtWidgets.QFormLayout()
        self.formLayout_23.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_23.setVerticalSpacing(15)
        self.formLayout_23.setObjectName("formLayout_23")
        spacerItem20 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_23.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem20)
        self.label_95 = QtWidgets.QLabel(self.widget_16)
        self.label_95.setObjectName("label_95")
        self.formLayout_23.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_95)
        self.horizontalLayout_68 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_68.setSpacing(8)
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.treesDead_1 = QtWidgets.QSpinBox(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_1.sizePolicy().hasHeightForWidth())
        self.treesDead_1.setSizePolicy(sizePolicy)
        self.treesDead_1.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_1.setMaximum(100)
        self.treesDead_1.setSingleStep(10)
        self.treesDead_1.setProperty("value", 1)
        self.treesDead_1.setObjectName("treesDead_1")
        self.horizontalLayout_68.addWidget(self.treesDead_1)
        self.label_96 = QtWidgets.QLabel(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_96.sizePolicy().hasHeightForWidth())
        self.label_96.setSizePolicy(sizePolicy)
        self.label_96.setObjectName("label_96")
        self.horizontalLayout_68.addWidget(self.label_96)
        self.formLayout_23.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_68
        )
        self.deadLeftText_1 = QtWidgets.QLabel(self.widget_16)
        self.deadLeftText_1.setObjectName("deadLeftText_1")
        self.formLayout_23.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_1
        )
        self.deadStemsLeftText_1 = QtWidgets.QLabel(self.widget_16)
        self.deadStemsLeftText_1.setObjectName("deadStemsLeftText_1")
        self.formLayout_23.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_1
        )
        self.horizontalLayout_69 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_69.setSpacing(8)
        self.horizontalLayout_69.setObjectName("horizontalLayout_69")
        self.deadStemsLeft_1 = QtWidgets.QSpinBox(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_1.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_1.setSizePolicy(sizePolicy)
        self.deadStemsLeft_1.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_1.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_1.setMaximum(100)
        self.deadStemsLeft_1.setSingleStep(10)
        self.deadStemsLeft_1.setObjectName("deadStemsLeft_1")
        self.horizontalLayout_69.addWidget(self.deadStemsLeft_1)
        self.deadStemsLeftLabel_1 = QtWidgets.QLabel(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_1.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_1.setObjectName("deadStemsLeftLabel_1")
        self.horizontalLayout_69.addWidget(self.deadStemsLeftLabel_1)
        self.formLayout_23.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_69
        )
        self.deadBranchesLeftText_1 = QtWidgets.QLabel(self.widget_16)
        self.deadBranchesLeftText_1.setObjectName("deadBranchesLeftText_1")
        self.formLayout_23.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_1
        )
        self.horizontalLayout_70 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_70.setSpacing(8)
        self.horizontalLayout_70.setObjectName("horizontalLayout_70")
        self.deadBranchesLeft_1 = QtWidgets.QSpinBox(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_1.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_1.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_1.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_1.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_1.setMaximum(100)
        self.deadBranchesLeft_1.setSingleStep(10)
        self.deadBranchesLeft_1.setObjectName("deadBranchesLeft_1")
        self.horizontalLayout_70.addWidget(self.deadBranchesLeft_1)
        self.deadBranchesLeftLabel_1 = QtWidgets.QLabel(self.widget_16)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_1.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_1.setObjectName("deadBranchesLeftLabel_1")
        self.horizontalLayout_70.addWidget(self.deadBranchesLeftLabel_1)
        self.formLayout_23.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_70
        )
        self.gridLayout_14.addLayout(self.formLayout_23, 1, 0, 1, 1)
        self.verticalLayout_45.addWidget(self.widget_16)
        self.stackedWidget.addWidget(self.treeMortPage_1)
        self.treeGrowthPage_1 = QtWidgets.QWidget()
        self.treeGrowthPage_1.setObjectName("treeGrowthPage_1")
        self.verticalLayout_47 = QtWidgets.QVBoxLayout(self.treeGrowthPage_1)
        self.verticalLayout_47.setObjectName("verticalLayout_47")
        self.widget_17 = QtWidgets.QWidget(self.treeGrowthPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_17.sizePolicy().hasHeightForWidth())
        self.widget_17.setSizePolicy(sizePolicy)
        self.widget_17.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_17.setObjectName("widget_17")
        self.gridLayout_19 = QtWidgets.QGridLayout(self.widget_17)
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.formLayout_24 = QtWidgets.QFormLayout()
        self.formLayout_24.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_24.setVerticalSpacing(15)
        self.formLayout_24.setObjectName("formLayout_24")
        spacerItem21 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_24.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem21)
        self.verticalLayout_46 = QtWidgets.QVBoxLayout()
        self.verticalLayout_46.setSpacing(8)
        self.verticalLayout_46.setObjectName("verticalLayout_46")
        self.growthCsvButton_1 = QtWidgets.QRadioButton(self.widget_17)
        self.growthCsvButton_1.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_1.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_1.setSizePolicy(sizePolicy)
        self.growthCsvButton_1.setObjectName("growthCsvButton_1")
        self.verticalLayout_46.addWidget(self.growthCsvButton_1)
        self.growthCustomButton_1 = QtWidgets.QRadioButton(self.widget_17)
        self.growthCustomButton_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_1.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_1.setSizePolicy(sizePolicy)
        self.growthCustomButton_1.setObjectName("growthCustomButton_1")
        self.verticalLayout_46.addWidget(self.growthCustomButton_1)
        self.formLayout_24.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_46
        )
        self.growthCsvLabel_1 = QtWidgets.QLabel(self.widget_17)
        self.growthCsvLabel_1.setEnabled(True)
        self.growthCsvLabel_1.setObjectName("growthCsvLabel_1")
        self.formLayout_24.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_1
        )
        self.horizontalLayout_71 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_71.setSpacing(8)
        self.horizontalLayout_71.setObjectName("horizontalLayout_71")
        self.growthCsvBrowseBox_1 = QtWidgets.QLineEdit(self.widget_17)
        self.growthCsvBrowseBox_1.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_1.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_1.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_1.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_1.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_1.setMaxLength(1000)
        self.growthCsvBrowseBox_1.setObjectName("growthCsvBrowseBox_1")
        self.horizontalLayout_71.addWidget(self.growthCsvBrowseBox_1)
        self.growthCsvBrowseButton_1 = QtWidgets.QPushButton(self.widget_17)
        self.growthCsvBrowseButton_1.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_1.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_1.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_1.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_1.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_1.setObjectName("growthCsvBrowseButton_1")
        self.horizontalLayout_71.addWidget(self.growthCsvBrowseButton_1)
        self.formLayout_24.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_71
        )
        self.label_97 = QtWidgets.QLabel(self.widget_17)
        self.label_97.setObjectName("label_97")
        self.formLayout_24.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_97)
        self.allom_1 = QtWidgets.QComboBox(self.widget_17)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_1.sizePolicy().hasHeightForWidth())
        self.allom_1.setSizePolicy(sizePolicy)
        self.allom_1.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_1.setObjectName("allom_1")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.setItemText(1, "")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.setItemText(7, "")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.allom_1.setItemText(12, "")
        self.allom_1.addItem("")
        self.allom_1.addItem("")
        self.formLayout_24.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_1)
        self.gridLayout_19.addLayout(self.formLayout_24, 1, 0, 1, 1)
        self.label_98 = QtWidgets.QLabel(self.widget_17)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_98.sizePolicy().hasHeightForWidth())
        self.label_98.setSizePolicy(sizePolicy)
        self.label_98.setMinimumSize(QtCore.QSize(0, 30))
        self.label_98.setObjectName("label_98")
        self.gridLayout_19.addWidget(self.label_98, 0, 0, 1, 1)
        self.verticalLayout_47.addWidget(self.widget_17)
        self.stackedWidget.addWidget(self.treeGrowthPage_1)
        self.treeMgmtPage_2 = QtWidgets.QWidget()
        self.treeMgmtPage_2.setObjectName("treeMgmtPage_2")
        self.verticalLayout_49 = QtWidgets.QVBoxLayout(self.treeMgmtPage_2)
        self.verticalLayout_49.setObjectName("verticalLayout_49")
        self.widget_18 = QtWidgets.QWidget(self.treeMgmtPage_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_18.sizePolicy().hasHeightForWidth())
        self.widget_18.setSizePolicy(sizePolicy)
        self.widget_18.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_18.setObjectName("widget_18")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.widget_18)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.label_99 = QtWidgets.QLabel(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_99.sizePolicy().hasHeightForWidth())
        self.label_99.setSizePolicy(sizePolicy)
        self.label_99.setMinimumSize(QtCore.QSize(0, 30))
        self.label_99.setObjectName("label_99")
        self.gridLayout_15.addWidget(self.label_99, 0, 0, 1, 1)
        self.formLayout_25 = QtWidgets.QFormLayout()
        self.formLayout_25.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_25.setVerticalSpacing(15)
        self.formLayout_25.setObjectName("formLayout_25")
        spacerItem22 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_25.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem22)
        self.label_100 = QtWidgets.QLabel(self.widget_18)
        self.label_100.setObjectName("label_100")
        self.formLayout_25.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_100)
        self.treeType_2 = QtWidgets.QComboBox(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_2.sizePolicy().hasHeightForWidth())
        self.treeType_2.setSizePolicy(sizePolicy)
        self.treeType_2.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_2.setObjectName("treeType_2")
        self.treeType_2.addItem("")
        self.treeType_2.addItem("")
        self.treeType_2.setItemText(1, "")
        self.treeType_2.addItem("")
        self.treeType_2.addItem("")
        self.treeType_2.addItem("")
        self.treeType_2.setItemText(4, "")
        self.treeType_2.addItem("")
        self.treeType_2.addItem("")
        self.formLayout_25.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_2
        )
        self.treeInputType_2 = QtWidgets.QLabel(self.widget_18)
        self.treeInputType_2.setEnabled(False)
        self.treeInputType_2.setObjectName("treeInputType_2")
        self.formLayout_25.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_2
        )
        self.verticalLayout_48 = QtWidgets.QVBoxLayout()
        self.verticalLayout_48.setSpacing(8)
        self.verticalLayout_48.setObjectName("verticalLayout_48")
        self.treeCsvButton_2 = QtWidgets.QRadioButton(self.widget_18)
        self.treeCsvButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_2.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_2.setSizePolicy(sizePolicy)
        self.treeCsvButton_2.setObjectName("treeCsvButton_2")
        self.verticalLayout_48.addWidget(self.treeCsvButton_2)
        self.treeCustomButton_2 = QtWidgets.QRadioButton(self.widget_18)
        self.treeCustomButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_2.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_2.setSizePolicy(sizePolicy)
        self.treeCustomButton_2.setObjectName("treeCustomButton_2")
        self.verticalLayout_48.addWidget(self.treeCustomButton_2)
        self.formLayout_25.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_48
        )
        self.treeCsvLabel_2 = QtWidgets.QLabel(self.widget_18)
        self.treeCsvLabel_2.setEnabled(False)
        self.treeCsvLabel_2.setObjectName("treeCsvLabel_2")
        self.formLayout_25.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_2
        )
        self.horizontalLayout_72 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_72.setSpacing(8)
        self.horizontalLayout_72.setObjectName("horizontalLayout_72")
        self.treeCsvBrowseBox_2 = QtWidgets.QLineEdit(self.widget_18)
        self.treeCsvBrowseBox_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_2.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_2.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_2.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_2.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_2.setMaxLength(1000)
        self.treeCsvBrowseBox_2.setObjectName("treeCsvBrowseBox_2")
        self.horizontalLayout_72.addWidget(self.treeCsvBrowseBox_2)
        self.treeCsvBrowseButton_2 = QtWidgets.QPushButton(self.widget_18)
        self.treeCsvBrowseButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_2.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_2.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_2.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_2.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_2.setObjectName("treeCsvBrowseButton_2")
        self.horizontalLayout_72.addWidget(self.treeCsvBrowseButton_2)
        self.formLayout_25.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_72
        )
        self.label_101 = QtWidgets.QLabel(self.widget_18)
        self.label_101.setObjectName("label_101")
        self.formLayout_25.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_101)
        self.horizontalLayout_73 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_73.setSpacing(8)
        self.horizontalLayout_73.setObjectName("horizontalLayout_73")
        self.standDens_2 = QtWidgets.QSpinBox(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_2.sizePolicy().hasHeightForWidth())
        self.standDens_2.setSizePolicy(sizePolicy)
        self.standDens_2.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_2.setMaximum(999)
        self.standDens_2.setSingleStep(50)
        self.standDens_2.setProperty("value", 200)
        self.standDens_2.setObjectName("standDens_2")
        self.horizontalLayout_73.addWidget(self.standDens_2)
        self.label_102 = QtWidgets.QLabel(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_102.sizePolicy().hasHeightForWidth())
        self.label_102.setSizePolicy(sizePolicy)
        self.label_102.setObjectName("label_102")
        self.horizontalLayout_73.addWidget(self.label_102)
        self.formLayout_25.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_73
        )
        self.label_19 = QtWidgets.QLabel(self.widget_18)
        self.label_19.setObjectName("label_19")
        self.formLayout_25.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.horizontalLayout_74 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_74.setSpacing(8)
        self.horizontalLayout_74.setObjectName("horizontalLayout_74")
        self.label_22 = QtWidgets.QLabel(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_74.addWidget(self.label_22)
        self.yearPlanted_2 = QtWidgets.QSpinBox(self.widget_18)
        self.yearPlanted_2.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_2.setObjectName("yearPlanted_2")
        self.horizontalLayout_74.addWidget(self.yearPlanted_2)
        self.formLayout_25.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_74
        )
        self.label_225 = QtWidgets.QLabel(self.widget_18)
        self.label_225.setObjectName("label_225")
        self.formLayout_25.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_225)
        self.speciesName_2 = QtWidgets.QLineEdit(self.widget_18)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_2.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_2.setSizePolicy(sizePolicy)
        self.speciesName_2.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_2.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_2.setObjectName("speciesName_2")
        self.formLayout_25.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_2
        )
        self.gridLayout_15.addLayout(self.formLayout_25, 1, 0, 1, 1)
        self.verticalLayout_49.addWidget(self.widget_18)
        self.stackedWidget.addWidget(self.treeMgmtPage_2)
        self.treeThinPage_2 = QtWidgets.QWidget()
        self.treeThinPage_2.setObjectName("treeThinPage_2")
        self.verticalLayout_51 = QtWidgets.QVBoxLayout(self.treeThinPage_2)
        self.verticalLayout_51.setObjectName("verticalLayout_51")
        self.widget_19 = QtWidgets.QWidget(self.treeThinPage_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_19.sizePolicy().hasHeightForWidth())
        self.widget_19.setSizePolicy(sizePolicy)
        self.widget_19.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_19.setObjectName("widget_19")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.widget_19)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.label_103 = QtWidgets.QLabel(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_103.sizePolicy().hasHeightForWidth())
        self.label_103.setSizePolicy(sizePolicy)
        self.label_103.setMinimumSize(QtCore.QSize(0, 30))
        self.label_103.setObjectName("label_103")
        self.gridLayout_16.addWidget(self.label_103, 0, 0, 1, 1)
        self.formLayout_26 = QtWidgets.QFormLayout()
        self.formLayout_26.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_26.setVerticalSpacing(15)
        self.formLayout_26.setObjectName("formLayout_26")
        spacerItem23 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_26.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem23)
        self.label_34 = QtWidgets.QLabel(self.widget_19)
        self.label_34.setObjectName("label_34")
        self.formLayout_26.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.verticalLayout_50 = QtWidgets.QVBoxLayout()
        self.verticalLayout_50.setObjectName("verticalLayout_50")
        self.thinNever_2 = QtWidgets.QRadioButton(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_2.sizePolicy().hasHeightForWidth())
        self.thinNever_2.setSizePolicy(sizePolicy)
        self.thinNever_2.setObjectName("thinNever_2")
        self.verticalLayout_50.addWidget(self.thinNever_2)
        self.horizontalLayout_76 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_76.setSpacing(8)
        self.horizontalLayout_76.setObjectName("horizontalLayout_76")
        self.thinInterval_2 = QtWidgets.QRadioButton(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_2.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_2.setSizePolicy(sizePolicy)
        self.thinInterval_2.setObjectName("thinInterval_2")
        self.horizontalLayout_76.addWidget(self.thinInterval_2)
        self.thinFreq_2 = QtWidgets.QSpinBox(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_2.sizePolicy().hasHeightForWidth())
        self.thinFreq_2.setSizePolicy(sizePolicy)
        self.thinFreq_2.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_2.setMinimum(1)
        self.thinFreq_2.setSingleStep(10)
        self.thinFreq_2.setProperty("value", 1)
        self.thinFreq_2.setObjectName("thinFreq_2")
        self.horizontalLayout_76.addWidget(self.thinFreq_2)
        self.thinFreqLabel_2 = QtWidgets.QLabel(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_2.setSizePolicy(sizePolicy)
        self.thinFreqLabel_2.setObjectName("thinFreqLabel_2")
        self.horizontalLayout_76.addWidget(self.thinFreqLabel_2)
        self.verticalLayout_50.addLayout(self.horizontalLayout_76)
        self.thinCustom_2 = QtWidgets.QRadioButton(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_2.sizePolicy().hasHeightForWidth())
        self.thinCustom_2.setSizePolicy(sizePolicy)
        self.thinCustom_2.setObjectName("thinCustom_2")
        self.verticalLayout_50.addWidget(self.thinCustom_2)
        self.formLayout_26.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_50
        )
        self.thinCustomText_2 = QtWidgets.QLabel(self.widget_19)
        self.thinCustomText_2.setObjectName("thinCustomText_2")
        self.formLayout_26.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_2
        )
        self.thinCustomLine_2 = QtWidgets.QLineEdit(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_2.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_2.setSizePolicy(sizePolicy)
        self.thinCustomLine_2.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_2.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_2.setObjectName("thinCustomLine_2")
        self.formLayout_26.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_2
        )
        self.treesThinnedText_2 = QtWidgets.QLabel(self.widget_19)
        self.treesThinnedText_2.setObjectName("treesThinnedText_2")
        self.formLayout_26.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_2
        )
        self.horizontalLayout_77 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_77.setSpacing(8)
        self.horizontalLayout_77.setObjectName("horizontalLayout_77")
        self.treesThinned_2 = QtWidgets.QSpinBox(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_2.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_2.setSizePolicy(sizePolicy)
        self.treesThinned_2.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_2.setMaximum(100)
        self.treesThinned_2.setSingleStep(10)
        self.treesThinned_2.setProperty("value", 50)
        self.treesThinned_2.setObjectName("treesThinned_2")
        self.horizontalLayout_77.addWidget(self.treesThinned_2)
        self.treesThinnedLabel_2 = QtWidgets.QLabel(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_2.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_2.setObjectName("treesThinnedLabel_2")
        self.horizontalLayout_77.addWidget(self.treesThinnedLabel_2)
        self.formLayout_26.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_77
        )
        self.thinnedLeftText_2 = QtWidgets.QLabel(self.widget_19)
        self.thinnedLeftText_2.setObjectName("thinnedLeftText_2")
        self.formLayout_26.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_2
        )
        self.thinnedStemsLeftText_2 = QtWidgets.QLabel(self.widget_19)
        self.thinnedStemsLeftText_2.setObjectName("thinnedStemsLeftText_2")
        self.formLayout_26.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_2
        )
        self.horizontalLayout_78 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_78.setSpacing(8)
        self.horizontalLayout_78.setObjectName("horizontalLayout_78")
        self.thinnedStemsLeft_2 = QtWidgets.QSpinBox(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_2.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_2.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_2.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_2.setMaximum(100)
        self.thinnedStemsLeft_2.setSingleStep(10)
        self.thinnedStemsLeft_2.setObjectName("thinnedStemsLeft_2")
        self.horizontalLayout_78.addWidget(self.thinnedStemsLeft_2)
        self.thinnedStemsLeftLabel_2 = QtWidgets.QLabel(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_2.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_2.setObjectName("thinnedStemsLeftLabel_2")
        self.horizontalLayout_78.addWidget(self.thinnedStemsLeftLabel_2)
        self.formLayout_26.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_78
        )
        self.thinnedBranchesLeftText_2 = QtWidgets.QLabel(self.widget_19)
        self.thinnedBranchesLeftText_2.setObjectName("thinnedBranchesLeftText_2")
        self.formLayout_26.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_2
        )
        self.horizontalLayout_79 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_79.setSpacing(8)
        self.horizontalLayout_79.setObjectName("horizontalLayout_79")
        self.thinnedBranchesLeft_2 = QtWidgets.QSpinBox(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_2.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_2.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_2.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_2.setMaximum(100)
        self.thinnedBranchesLeft_2.setSingleStep(10)
        self.thinnedBranchesLeft_2.setObjectName("thinnedBranchesLeft_2")
        self.horizontalLayout_79.addWidget(self.thinnedBranchesLeft_2)
        self.thinnedBranchesLeftLabel_2 = QtWidgets.QLabel(self.widget_19)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_2.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_2.setObjectName("thinnedBranchesLeftLabel_2")
        self.horizontalLayout_79.addWidget(self.thinnedBranchesLeftLabel_2)
        self.formLayout_26.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_79
        )
        self.gridLayout_16.addLayout(self.formLayout_26, 1, 0, 1, 1)
        self.verticalLayout_51.addWidget(self.widget_19)
        self.stackedWidget.addWidget(self.treeThinPage_2)
        self.treeMortPage_2 = QtWidgets.QWidget()
        self.treeMortPage_2.setObjectName("treeMortPage_2")
        self.verticalLayout_52 = QtWidgets.QVBoxLayout(self.treeMortPage_2)
        self.verticalLayout_52.setObjectName("verticalLayout_52")
        self.widget_20 = QtWidgets.QWidget(self.treeMortPage_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_20.sizePolicy().hasHeightForWidth())
        self.widget_20.setSizePolicy(sizePolicy)
        self.widget_20.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_20.setObjectName("widget_20")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.widget_20)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.label_104 = QtWidgets.QLabel(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_104.sizePolicy().hasHeightForWidth())
        self.label_104.setSizePolicy(sizePolicy)
        self.label_104.setMinimumSize(QtCore.QSize(0, 30))
        self.label_104.setObjectName("label_104")
        self.gridLayout_17.addWidget(self.label_104, 0, 0, 1, 1)
        self.formLayout_27 = QtWidgets.QFormLayout()
        self.formLayout_27.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_27.setVerticalSpacing(15)
        self.formLayout_27.setObjectName("formLayout_27")
        spacerItem24 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_27.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem24)
        self.label_105 = QtWidgets.QLabel(self.widget_20)
        self.label_105.setObjectName("label_105")
        self.formLayout_27.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_105)
        self.horizontalLayout_80 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_80.setSpacing(8)
        self.horizontalLayout_80.setObjectName("horizontalLayout_80")
        self.treesDead_2 = QtWidgets.QSpinBox(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_2.sizePolicy().hasHeightForWidth())
        self.treesDead_2.setSizePolicy(sizePolicy)
        self.treesDead_2.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_2.setMaximum(100)
        self.treesDead_2.setSingleStep(10)
        self.treesDead_2.setProperty("value", 1)
        self.treesDead_2.setObjectName("treesDead_2")
        self.horizontalLayout_80.addWidget(self.treesDead_2)
        self.label_106 = QtWidgets.QLabel(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_106.sizePolicy().hasHeightForWidth())
        self.label_106.setSizePolicy(sizePolicy)
        self.label_106.setObjectName("label_106")
        self.horizontalLayout_80.addWidget(self.label_106)
        self.formLayout_27.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_80
        )
        self.deadLeftText_2 = QtWidgets.QLabel(self.widget_20)
        self.deadLeftText_2.setObjectName("deadLeftText_2")
        self.formLayout_27.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_2
        )
        self.deadStemsLeftText_2 = QtWidgets.QLabel(self.widget_20)
        self.deadStemsLeftText_2.setObjectName("deadStemsLeftText_2")
        self.formLayout_27.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_2
        )
        self.horizontalLayout_81 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_81.setSpacing(8)
        self.horizontalLayout_81.setObjectName("horizontalLayout_81")
        self.deadStemsLeft_2 = QtWidgets.QSpinBox(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_2.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_2.setSizePolicy(sizePolicy)
        self.deadStemsLeft_2.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_2.setMaximum(100)
        self.deadStemsLeft_2.setSingleStep(10)
        self.deadStemsLeft_2.setObjectName("deadStemsLeft_2")
        self.horizontalLayout_81.addWidget(self.deadStemsLeft_2)
        self.deadStemsLeftLabel_2 = QtWidgets.QLabel(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_2.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_2.setObjectName("deadStemsLeftLabel_2")
        self.horizontalLayout_81.addWidget(self.deadStemsLeftLabel_2)
        self.formLayout_27.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_81
        )
        self.deadBranchesLeftText_2 = QtWidgets.QLabel(self.widget_20)
        self.deadBranchesLeftText_2.setObjectName("deadBranchesLeftText_2")
        self.formLayout_27.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_2
        )
        self.horizontalLayout_82 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_82.setSpacing(8)
        self.horizontalLayout_82.setObjectName("horizontalLayout_82")
        self.deadBranchesLeft_2 = QtWidgets.QSpinBox(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_2.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_2.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_2.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_2.setMaximum(100)
        self.deadBranchesLeft_2.setSingleStep(10)
        self.deadBranchesLeft_2.setObjectName("deadBranchesLeft_2")
        self.horizontalLayout_82.addWidget(self.deadBranchesLeft_2)
        self.deadBranchesLeftLabel_2 = QtWidgets.QLabel(self.widget_20)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_2.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_2.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_2.setObjectName("deadBranchesLeftLabel_2")
        self.horizontalLayout_82.addWidget(self.deadBranchesLeftLabel_2)
        self.formLayout_27.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_82
        )
        self.gridLayout_17.addLayout(self.formLayout_27, 1, 0, 1, 1)
        self.verticalLayout_52.addWidget(self.widget_20)
        self.stackedWidget.addWidget(self.treeMortPage_2)
        self.treeGrowthPage_2 = QtWidgets.QWidget()
        self.treeGrowthPage_2.setObjectName("treeGrowthPage_2")
        self.verticalLayout_54 = QtWidgets.QVBoxLayout(self.treeGrowthPage_2)
        self.verticalLayout_54.setObjectName("verticalLayout_54")
        self.widget_21 = QtWidgets.QWidget(self.treeGrowthPage_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_21.sizePolicy().hasHeightForWidth())
        self.widget_21.setSizePolicy(sizePolicy)
        self.widget_21.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_21.setObjectName("widget_21")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.widget_21)
        self.gridLayout_21.setObjectName("gridLayout_21")
        self.formLayout_28 = QtWidgets.QFormLayout()
        self.formLayout_28.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_28.setVerticalSpacing(15)
        self.formLayout_28.setObjectName("formLayout_28")
        spacerItem25 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_28.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem25)
        self.verticalLayout_53 = QtWidgets.QVBoxLayout()
        self.verticalLayout_53.setSpacing(8)
        self.verticalLayout_53.setObjectName("verticalLayout_53")
        self.growthCsvButton_2 = QtWidgets.QRadioButton(self.widget_21)
        self.growthCsvButton_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_2.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_2.setSizePolicy(sizePolicy)
        self.growthCsvButton_2.setObjectName("growthCsvButton_2")
        self.verticalLayout_53.addWidget(self.growthCsvButton_2)
        self.growthCustomButton_2 = QtWidgets.QRadioButton(self.widget_21)
        self.growthCustomButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_2.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_2.setSizePolicy(sizePolicy)
        self.growthCustomButton_2.setObjectName("growthCustomButton_2")
        self.verticalLayout_53.addWidget(self.growthCustomButton_2)
        self.formLayout_28.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_53
        )
        self.growthCsvLabel_2 = QtWidgets.QLabel(self.widget_21)
        self.growthCsvLabel_2.setEnabled(True)
        self.growthCsvLabel_2.setObjectName("growthCsvLabel_2")
        self.formLayout_28.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_2
        )
        self.horizontalLayout_83 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_83.setSpacing(8)
        self.horizontalLayout_83.setObjectName("horizontalLayout_83")
        self.growthCsvBrowseBox_2 = QtWidgets.QLineEdit(self.widget_21)
        self.growthCsvBrowseBox_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_2.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_2.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_2.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_2.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_2.setMaxLength(1000)
        self.growthCsvBrowseBox_2.setObjectName("growthCsvBrowseBox_2")
        self.horizontalLayout_83.addWidget(self.growthCsvBrowseBox_2)
        self.growthCsvBrowseButton_2 = QtWidgets.QPushButton(self.widget_21)
        self.growthCsvBrowseButton_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_2.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_2.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_2.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_2.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_2.setObjectName("growthCsvBrowseButton_2")
        self.horizontalLayout_83.addWidget(self.growthCsvBrowseButton_2)
        self.formLayout_28.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_83
        )
        self.label_107 = QtWidgets.QLabel(self.widget_21)
        self.label_107.setObjectName("label_107")
        self.formLayout_28.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_107)
        self.allom_2 = QtWidgets.QComboBox(self.widget_21)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_2.sizePolicy().hasHeightForWidth())
        self.allom_2.setSizePolicy(sizePolicy)
        self.allom_2.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_2.setObjectName("allom_2")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.setItemText(1, "")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.setItemText(7, "")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.allom_2.setItemText(12, "")
        self.allom_2.addItem("")
        self.allom_2.addItem("")
        self.formLayout_28.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_2)
        self.gridLayout_21.addLayout(self.formLayout_28, 1, 0, 1, 1)
        self.label_108 = QtWidgets.QLabel(self.widget_21)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_108.sizePolicy().hasHeightForWidth())
        self.label_108.setSizePolicy(sizePolicy)
        self.label_108.setMinimumSize(QtCore.QSize(0, 30))
        self.label_108.setObjectName("label_108")
        self.gridLayout_21.addWidget(self.label_108, 0, 0, 1, 1)
        self.verticalLayout_54.addWidget(self.widget_21)
        self.stackedWidget.addWidget(self.treeGrowthPage_2)
        self.treeMgmtPage_3 = QtWidgets.QWidget()
        self.treeMgmtPage_3.setObjectName("treeMgmtPage_3")
        self.verticalLayout_56 = QtWidgets.QVBoxLayout(self.treeMgmtPage_3)
        self.verticalLayout_56.setObjectName("verticalLayout_56")
        self.widget_22 = QtWidgets.QWidget(self.treeMgmtPage_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_22.sizePolicy().hasHeightForWidth())
        self.widget_22.setSizePolicy(sizePolicy)
        self.widget_22.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_22.setObjectName("widget_22")
        self.gridLayout_22 = QtWidgets.QGridLayout(self.widget_22)
        self.gridLayout_22.setObjectName("gridLayout_22")
        self.label_109 = QtWidgets.QLabel(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_109.sizePolicy().hasHeightForWidth())
        self.label_109.setSizePolicy(sizePolicy)
        self.label_109.setMinimumSize(QtCore.QSize(0, 30))
        self.label_109.setObjectName("label_109")
        self.gridLayout_22.addWidget(self.label_109, 0, 0, 1, 1)
        self.formLayout_29 = QtWidgets.QFormLayout()
        self.formLayout_29.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_29.setVerticalSpacing(15)
        self.formLayout_29.setObjectName("formLayout_29")
        spacerItem26 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_29.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem26)
        self.label_110 = QtWidgets.QLabel(self.widget_22)
        self.label_110.setObjectName("label_110")
        self.formLayout_29.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_110)
        self.treeType_3 = QtWidgets.QComboBox(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_3.sizePolicy().hasHeightForWidth())
        self.treeType_3.setSizePolicy(sizePolicy)
        self.treeType_3.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_3.setObjectName("treeType_3")
        self.treeType_3.addItem("")
        self.treeType_3.addItem("")
        self.treeType_3.setItemText(1, "")
        self.treeType_3.addItem("")
        self.treeType_3.addItem("")
        self.treeType_3.addItem("")
        self.treeType_3.setItemText(4, "")
        self.treeType_3.addItem("")
        self.treeType_3.addItem("")
        self.formLayout_29.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_3
        )
        self.treeInputType_3 = QtWidgets.QLabel(self.widget_22)
        self.treeInputType_3.setEnabled(False)
        self.treeInputType_3.setObjectName("treeInputType_3")
        self.formLayout_29.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_3
        )
        self.verticalLayout_55 = QtWidgets.QVBoxLayout()
        self.verticalLayout_55.setSpacing(8)
        self.verticalLayout_55.setObjectName("verticalLayout_55")
        self.treeCsvButton_3 = QtWidgets.QRadioButton(self.widget_22)
        self.treeCsvButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_3.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_3.setSizePolicy(sizePolicy)
        self.treeCsvButton_3.setObjectName("treeCsvButton_3")
        self.verticalLayout_55.addWidget(self.treeCsvButton_3)
        self.treeCustomButton_3 = QtWidgets.QRadioButton(self.widget_22)
        self.treeCustomButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_3.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_3.setSizePolicy(sizePolicy)
        self.treeCustomButton_3.setObjectName("treeCustomButton_3")
        self.verticalLayout_55.addWidget(self.treeCustomButton_3)
        self.formLayout_29.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_55
        )
        self.treeCsvLabel_3 = QtWidgets.QLabel(self.widget_22)
        self.treeCsvLabel_3.setEnabled(False)
        self.treeCsvLabel_3.setObjectName("treeCsvLabel_3")
        self.formLayout_29.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_3
        )
        self.horizontalLayout_84 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_84.setSpacing(8)
        self.horizontalLayout_84.setObjectName("horizontalLayout_84")
        self.treeCsvBrowseBox_3 = QtWidgets.QLineEdit(self.widget_22)
        self.treeCsvBrowseBox_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_3.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_3.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_3.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_3.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_3.setMaxLength(1000)
        self.treeCsvBrowseBox_3.setObjectName("treeCsvBrowseBox_3")
        self.horizontalLayout_84.addWidget(self.treeCsvBrowseBox_3)
        self.treeCsvBrowseButton_3 = QtWidgets.QPushButton(self.widget_22)
        self.treeCsvBrowseButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_3.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_3.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_3.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_3.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_3.setObjectName("treeCsvBrowseButton_3")
        self.horizontalLayout_84.addWidget(self.treeCsvBrowseButton_3)
        self.formLayout_29.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_84
        )
        self.label_111 = QtWidgets.QLabel(self.widget_22)
        self.label_111.setObjectName("label_111")
        self.formLayout_29.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_111)
        self.horizontalLayout_85 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_85.setSpacing(8)
        self.horizontalLayout_85.setObjectName("horizontalLayout_85")
        self.standDens_3 = QtWidgets.QSpinBox(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_3.sizePolicy().hasHeightForWidth())
        self.standDens_3.setSizePolicy(sizePolicy)
        self.standDens_3.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_3.setMaximum(999)
        self.standDens_3.setSingleStep(50)
        self.standDens_3.setProperty("value", 200)
        self.standDens_3.setObjectName("standDens_3")
        self.horizontalLayout_85.addWidget(self.standDens_3)
        self.label_112 = QtWidgets.QLabel(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_112.sizePolicy().hasHeightForWidth())
        self.label_112.setSizePolicy(sizePolicy)
        self.label_112.setObjectName("label_112")
        self.horizontalLayout_85.addWidget(self.label_112)
        self.formLayout_29.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_85
        )
        self.label_23 = QtWidgets.QLabel(self.widget_22)
        self.label_23.setObjectName("label_23")
        self.formLayout_29.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.horizontalLayout_86 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_86.setSpacing(8)
        self.horizontalLayout_86.setObjectName("horizontalLayout_86")
        self.label_24 = QtWidgets.QLabel(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_86.addWidget(self.label_24)
        self.yearPlanted_3 = QtWidgets.QSpinBox(self.widget_22)
        self.yearPlanted_3.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_3.setObjectName("yearPlanted_3")
        self.horizontalLayout_86.addWidget(self.yearPlanted_3)
        self.formLayout_29.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_86
        )
        self.label_224 = QtWidgets.QLabel(self.widget_22)
        self.label_224.setObjectName("label_224")
        self.formLayout_29.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_224)
        self.speciesName_3 = QtWidgets.QLineEdit(self.widget_22)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_3.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_3.setSizePolicy(sizePolicy)
        self.speciesName_3.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_3.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_3.setObjectName("speciesName_3")
        self.formLayout_29.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_3
        )
        self.gridLayout_22.addLayout(self.formLayout_29, 1, 0, 1, 1)
        self.verticalLayout_56.addWidget(self.widget_22)
        self.stackedWidget.addWidget(self.treeMgmtPage_3)
        self.treeThinPage_3 = QtWidgets.QWidget()
        self.treeThinPage_3.setObjectName("treeThinPage_3")
        self.verticalLayout_58 = QtWidgets.QVBoxLayout(self.treeThinPage_3)
        self.verticalLayout_58.setObjectName("verticalLayout_58")
        self.widget_23 = QtWidgets.QWidget(self.treeThinPage_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_23.sizePolicy().hasHeightForWidth())
        self.widget_23.setSizePolicy(sizePolicy)
        self.widget_23.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_23.setObjectName("widget_23")
        self.gridLayout_23 = QtWidgets.QGridLayout(self.widget_23)
        self.gridLayout_23.setObjectName("gridLayout_23")
        self.label_113 = QtWidgets.QLabel(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_113.sizePolicy().hasHeightForWidth())
        self.label_113.setSizePolicy(sizePolicy)
        self.label_113.setMinimumSize(QtCore.QSize(0, 30))
        self.label_113.setObjectName("label_113")
        self.gridLayout_23.addWidget(self.label_113, 0, 0, 1, 1)
        self.formLayout_30 = QtWidgets.QFormLayout()
        self.formLayout_30.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_30.setVerticalSpacing(15)
        self.formLayout_30.setObjectName("formLayout_30")
        spacerItem27 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_30.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem27)
        self.label_35 = QtWidgets.QLabel(self.widget_23)
        self.label_35.setObjectName("label_35")
        self.formLayout_30.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_35)
        self.verticalLayout_57 = QtWidgets.QVBoxLayout()
        self.verticalLayout_57.setObjectName("verticalLayout_57")
        self.thinNever_3 = QtWidgets.QRadioButton(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_3.sizePolicy().hasHeightForWidth())
        self.thinNever_3.setSizePolicy(sizePolicy)
        self.thinNever_3.setObjectName("thinNever_3")
        self.verticalLayout_57.addWidget(self.thinNever_3)
        self.horizontalLayout_88 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_88.setSpacing(8)
        self.horizontalLayout_88.setObjectName("horizontalLayout_88")
        self.thinInterval_3 = QtWidgets.QRadioButton(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_3.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_3.setSizePolicy(sizePolicy)
        self.thinInterval_3.setObjectName("thinInterval_3")
        self.horizontalLayout_88.addWidget(self.thinInterval_3)
        self.thinFreq_3 = QtWidgets.QSpinBox(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_3.sizePolicy().hasHeightForWidth())
        self.thinFreq_3.setSizePolicy(sizePolicy)
        self.thinFreq_3.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_3.setMinimum(1)
        self.thinFreq_3.setSingleStep(10)
        self.thinFreq_3.setProperty("value", 1)
        self.thinFreq_3.setObjectName("thinFreq_3")
        self.horizontalLayout_88.addWidget(self.thinFreq_3)
        self.thinFreqLabel_3 = QtWidgets.QLabel(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_3.setSizePolicy(sizePolicy)
        self.thinFreqLabel_3.setObjectName("thinFreqLabel_3")
        self.horizontalLayout_88.addWidget(self.thinFreqLabel_3)
        self.verticalLayout_57.addLayout(self.horizontalLayout_88)
        self.thinCustom_3 = QtWidgets.QRadioButton(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_3.sizePolicy().hasHeightForWidth())
        self.thinCustom_3.setSizePolicy(sizePolicy)
        self.thinCustom_3.setObjectName("thinCustom_3")
        self.verticalLayout_57.addWidget(self.thinCustom_3)
        self.formLayout_30.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_57
        )
        self.thinCustomText_3 = QtWidgets.QLabel(self.widget_23)
        self.thinCustomText_3.setObjectName("thinCustomText_3")
        self.formLayout_30.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_3
        )
        self.thinCustomLine_3 = QtWidgets.QLineEdit(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_3.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_3.setSizePolicy(sizePolicy)
        self.thinCustomLine_3.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_3.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_3.setObjectName("thinCustomLine_3")
        self.formLayout_30.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_3
        )
        self.treesThinnedText_3 = QtWidgets.QLabel(self.widget_23)
        self.treesThinnedText_3.setObjectName("treesThinnedText_3")
        self.formLayout_30.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_3
        )
        self.horizontalLayout_89 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_89.setSpacing(8)
        self.horizontalLayout_89.setObjectName("horizontalLayout_89")
        self.treesThinned_3 = QtWidgets.QSpinBox(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_3.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_3.setSizePolicy(sizePolicy)
        self.treesThinned_3.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_3.setMaximum(100)
        self.treesThinned_3.setSingleStep(10)
        self.treesThinned_3.setProperty("value", 50)
        self.treesThinned_3.setObjectName("treesThinned_3")
        self.horizontalLayout_89.addWidget(self.treesThinned_3)
        self.treesThinnedLabel_3 = QtWidgets.QLabel(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_3.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_3.setObjectName("treesThinnedLabel_3")
        self.horizontalLayout_89.addWidget(self.treesThinnedLabel_3)
        self.formLayout_30.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_89
        )
        self.thinnedLeftText_3 = QtWidgets.QLabel(self.widget_23)
        self.thinnedLeftText_3.setObjectName("thinnedLeftText_3")
        self.formLayout_30.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_3
        )
        self.thinnedStemsLeftText_3 = QtWidgets.QLabel(self.widget_23)
        self.thinnedStemsLeftText_3.setObjectName("thinnedStemsLeftText_3")
        self.formLayout_30.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_3
        )
        self.horizontalLayout_90 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_90.setSpacing(8)
        self.horizontalLayout_90.setObjectName("horizontalLayout_90")
        self.thinnedStemsLeft_3 = QtWidgets.QSpinBox(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_3.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_3.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_3.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_3.setMaximum(100)
        self.thinnedStemsLeft_3.setSingleStep(10)
        self.thinnedStemsLeft_3.setObjectName("thinnedStemsLeft_3")
        self.horizontalLayout_90.addWidget(self.thinnedStemsLeft_3)
        self.thinnedStemsLeftLabel_3 = QtWidgets.QLabel(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_3.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_3.setObjectName("thinnedStemsLeftLabel_3")
        self.horizontalLayout_90.addWidget(self.thinnedStemsLeftLabel_3)
        self.formLayout_30.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_90
        )
        self.thinnedBranchesLeftText_3 = QtWidgets.QLabel(self.widget_23)
        self.thinnedBranchesLeftText_3.setObjectName("thinnedBranchesLeftText_3")
        self.formLayout_30.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_3
        )
        self.horizontalLayout_91 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_91.setSpacing(8)
        self.horizontalLayout_91.setObjectName("horizontalLayout_91")
        self.thinnedBranchesLeft_3 = QtWidgets.QSpinBox(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_3.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_3.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_3.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_3.setMaximum(100)
        self.thinnedBranchesLeft_3.setSingleStep(10)
        self.thinnedBranchesLeft_3.setObjectName("thinnedBranchesLeft_3")
        self.horizontalLayout_91.addWidget(self.thinnedBranchesLeft_3)
        self.thinnedBranchesLeftLabel_3 = QtWidgets.QLabel(self.widget_23)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_3.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_3.setObjectName("thinnedBranchesLeftLabel_3")
        self.horizontalLayout_91.addWidget(self.thinnedBranchesLeftLabel_3)
        self.formLayout_30.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_91
        )
        self.gridLayout_23.addLayout(self.formLayout_30, 1, 0, 1, 1)
        self.verticalLayout_58.addWidget(self.widget_23)
        self.stackedWidget.addWidget(self.treeThinPage_3)
        self.treeMortPage_3 = QtWidgets.QWidget()
        self.treeMortPage_3.setObjectName("treeMortPage_3")
        self.verticalLayout_59 = QtWidgets.QVBoxLayout(self.treeMortPage_3)
        self.verticalLayout_59.setObjectName("verticalLayout_59")
        self.widget_24 = QtWidgets.QWidget(self.treeMortPage_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_24.sizePolicy().hasHeightForWidth())
        self.widget_24.setSizePolicy(sizePolicy)
        self.widget_24.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_24.setObjectName("widget_24")
        self.gridLayout_24 = QtWidgets.QGridLayout(self.widget_24)
        self.gridLayout_24.setObjectName("gridLayout_24")
        self.label_114 = QtWidgets.QLabel(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_114.sizePolicy().hasHeightForWidth())
        self.label_114.setSizePolicy(sizePolicy)
        self.label_114.setMinimumSize(QtCore.QSize(0, 30))
        self.label_114.setObjectName("label_114")
        self.gridLayout_24.addWidget(self.label_114, 0, 0, 1, 1)
        self.formLayout_31 = QtWidgets.QFormLayout()
        self.formLayout_31.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_31.setVerticalSpacing(15)
        self.formLayout_31.setObjectName("formLayout_31")
        spacerItem28 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_31.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem28)
        self.label_115 = QtWidgets.QLabel(self.widget_24)
        self.label_115.setObjectName("label_115")
        self.formLayout_31.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_115)
        self.horizontalLayout_92 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_92.setSpacing(8)
        self.horizontalLayout_92.setObjectName("horizontalLayout_92")
        self.treesDead_3 = QtWidgets.QSpinBox(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_3.sizePolicy().hasHeightForWidth())
        self.treesDead_3.setSizePolicy(sizePolicy)
        self.treesDead_3.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_3.setMaximum(100)
        self.treesDead_3.setSingleStep(10)
        self.treesDead_3.setProperty("value", 1)
        self.treesDead_3.setObjectName("treesDead_3")
        self.horizontalLayout_92.addWidget(self.treesDead_3)
        self.label_116 = QtWidgets.QLabel(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_116.sizePolicy().hasHeightForWidth())
        self.label_116.setSizePolicy(sizePolicy)
        self.label_116.setObjectName("label_116")
        self.horizontalLayout_92.addWidget(self.label_116)
        self.formLayout_31.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_92
        )
        self.deadLeftText_3 = QtWidgets.QLabel(self.widget_24)
        self.deadLeftText_3.setObjectName("deadLeftText_3")
        self.formLayout_31.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_3
        )
        self.deadStemsLeftText_3 = QtWidgets.QLabel(self.widget_24)
        self.deadStemsLeftText_3.setObjectName("deadStemsLeftText_3")
        self.formLayout_31.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_3
        )
        self.horizontalLayout_93 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_93.setSpacing(8)
        self.horizontalLayout_93.setObjectName("horizontalLayout_93")
        self.deadStemsLeft_3 = QtWidgets.QSpinBox(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_3.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_3.setSizePolicy(sizePolicy)
        self.deadStemsLeft_3.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_3.setMaximum(100)
        self.deadStemsLeft_3.setSingleStep(10)
        self.deadStemsLeft_3.setObjectName("deadStemsLeft_3")
        self.horizontalLayout_93.addWidget(self.deadStemsLeft_3)
        self.deadStemsLeftLabel_3 = QtWidgets.QLabel(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_3.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_3.setObjectName("deadStemsLeftLabel_3")
        self.horizontalLayout_93.addWidget(self.deadStemsLeftLabel_3)
        self.formLayout_31.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_93
        )
        self.deadBranchesLeftText_3 = QtWidgets.QLabel(self.widget_24)
        self.deadBranchesLeftText_3.setObjectName("deadBranchesLeftText_3")
        self.formLayout_31.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_3
        )
        self.horizontalLayout_94 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_94.setSpacing(8)
        self.horizontalLayout_94.setObjectName("horizontalLayout_94")
        self.deadBranchesLeft_3 = QtWidgets.QSpinBox(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_3.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_3.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_3.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_3.setMaximum(100)
        self.deadBranchesLeft_3.setSingleStep(10)
        self.deadBranchesLeft_3.setObjectName("deadBranchesLeft_3")
        self.horizontalLayout_94.addWidget(self.deadBranchesLeft_3)
        self.deadBranchesLeftLabel_3 = QtWidgets.QLabel(self.widget_24)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_3.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_3.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_3.setObjectName("deadBranchesLeftLabel_3")
        self.horizontalLayout_94.addWidget(self.deadBranchesLeftLabel_3)
        self.formLayout_31.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_94
        )
        self.gridLayout_24.addLayout(self.formLayout_31, 1, 0, 1, 1)
        self.verticalLayout_59.addWidget(self.widget_24)
        self.stackedWidget.addWidget(self.treeMortPage_3)
        self.treeGrowthPage_3 = QtWidgets.QWidget()
        self.treeGrowthPage_3.setObjectName("treeGrowthPage_3")
        self.verticalLayout_61 = QtWidgets.QVBoxLayout(self.treeGrowthPage_3)
        self.verticalLayout_61.setObjectName("verticalLayout_61")
        self.widget_25 = QtWidgets.QWidget(self.treeGrowthPage_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_25.sizePolicy().hasHeightForWidth())
        self.widget_25.setSizePolicy(sizePolicy)
        self.widget_25.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_25.setObjectName("widget_25")
        self.gridLayout_25 = QtWidgets.QGridLayout(self.widget_25)
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.formLayout_32 = QtWidgets.QFormLayout()
        self.formLayout_32.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_32.setVerticalSpacing(15)
        self.formLayout_32.setObjectName("formLayout_32")
        spacerItem29 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_32.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem29)
        self.verticalLayout_60 = QtWidgets.QVBoxLayout()
        self.verticalLayout_60.setSpacing(8)
        self.verticalLayout_60.setObjectName("verticalLayout_60")
        self.growthCsvButton_3 = QtWidgets.QRadioButton(self.widget_25)
        self.growthCsvButton_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_3.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_3.setSizePolicy(sizePolicy)
        self.growthCsvButton_3.setObjectName("growthCsvButton_3")
        self.verticalLayout_60.addWidget(self.growthCsvButton_3)
        self.growthCustomButton_3 = QtWidgets.QRadioButton(self.widget_25)
        self.growthCustomButton_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_3.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_3.setSizePolicy(sizePolicy)
        self.growthCustomButton_3.setObjectName("growthCustomButton_3")
        self.verticalLayout_60.addWidget(self.growthCustomButton_3)
        self.formLayout_32.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_60
        )
        self.growthCsvLabel_3 = QtWidgets.QLabel(self.widget_25)
        self.growthCsvLabel_3.setEnabled(True)
        self.growthCsvLabel_3.setObjectName("growthCsvLabel_3")
        self.formLayout_32.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_3
        )
        self.horizontalLayout_95 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_95.setSpacing(8)
        self.horizontalLayout_95.setObjectName("horizontalLayout_95")
        self.growthCsvBrowseBox_3 = QtWidgets.QLineEdit(self.widget_25)
        self.growthCsvBrowseBox_3.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_3.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_3.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_3.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_3.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_3.setMaxLength(1000)
        self.growthCsvBrowseBox_3.setObjectName("growthCsvBrowseBox_3")
        self.horizontalLayout_95.addWidget(self.growthCsvBrowseBox_3)
        self.growthCsvBrowseButton_3 = QtWidgets.QPushButton(self.widget_25)
        self.growthCsvBrowseButton_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_3.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_3.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_3.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_3.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_3.setObjectName("growthCsvBrowseButton_3")
        self.horizontalLayout_95.addWidget(self.growthCsvBrowseButton_3)
        self.formLayout_32.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_95
        )
        self.label_117 = QtWidgets.QLabel(self.widget_25)
        self.label_117.setObjectName("label_117")
        self.formLayout_32.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_117)
        self.allom_3 = QtWidgets.QComboBox(self.widget_25)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_3.sizePolicy().hasHeightForWidth())
        self.allom_3.setSizePolicy(sizePolicy)
        self.allom_3.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_3.setObjectName("allom_3")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.setItemText(1, "")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.setItemText(7, "")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.allom_3.setItemText(12, "")
        self.allom_3.addItem("")
        self.allom_3.addItem("")
        self.formLayout_32.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_3)
        self.gridLayout_25.addLayout(self.formLayout_32, 1, 0, 1, 1)
        self.label_118 = QtWidgets.QLabel(self.widget_25)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_118.sizePolicy().hasHeightForWidth())
        self.label_118.setSizePolicy(sizePolicy)
        self.label_118.setMinimumSize(QtCore.QSize(0, 30))
        self.label_118.setObjectName("label_118")
        self.gridLayout_25.addWidget(self.label_118, 0, 0, 1, 1)
        self.verticalLayout_61.addWidget(self.widget_25)
        self.stackedWidget.addWidget(self.treeGrowthPage_3)
        self.treeMgmtPage_4 = QtWidgets.QWidget()
        self.treeMgmtPage_4.setObjectName("treeMgmtPage_4")
        self.verticalLayout_63 = QtWidgets.QVBoxLayout(self.treeMgmtPage_4)
        self.verticalLayout_63.setObjectName("verticalLayout_63")
        self.widget_26 = QtWidgets.QWidget(self.treeMgmtPage_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_26.sizePolicy().hasHeightForWidth())
        self.widget_26.setSizePolicy(sizePolicy)
        self.widget_26.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_26.setObjectName("widget_26")
        self.gridLayout_26 = QtWidgets.QGridLayout(self.widget_26)
        self.gridLayout_26.setObjectName("gridLayout_26")
        self.label_119 = QtWidgets.QLabel(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_119.sizePolicy().hasHeightForWidth())
        self.label_119.setSizePolicy(sizePolicy)
        self.label_119.setMinimumSize(QtCore.QSize(0, 30))
        self.label_119.setObjectName("label_119")
        self.gridLayout_26.addWidget(self.label_119, 0, 0, 1, 1)
        self.formLayout_33 = QtWidgets.QFormLayout()
        self.formLayout_33.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_33.setVerticalSpacing(15)
        self.formLayout_33.setObjectName("formLayout_33")
        spacerItem30 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_33.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem30)
        self.label_120 = QtWidgets.QLabel(self.widget_26)
        self.label_120.setObjectName("label_120")
        self.formLayout_33.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_120)
        self.treeType_4 = QtWidgets.QComboBox(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_4.sizePolicy().hasHeightForWidth())
        self.treeType_4.setSizePolicy(sizePolicy)
        self.treeType_4.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_4.setObjectName("treeType_4")
        self.treeType_4.addItem("")
        self.treeType_4.addItem("")
        self.treeType_4.setItemText(1, "")
        self.treeType_4.addItem("")
        self.treeType_4.addItem("")
        self.treeType_4.addItem("")
        self.treeType_4.setItemText(4, "")
        self.treeType_4.addItem("")
        self.treeType_4.addItem("")
        self.formLayout_33.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_4
        )
        self.treeInputType_4 = QtWidgets.QLabel(self.widget_26)
        self.treeInputType_4.setEnabled(False)
        self.treeInputType_4.setObjectName("treeInputType_4")
        self.formLayout_33.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_4
        )
        self.verticalLayout_62 = QtWidgets.QVBoxLayout()
        self.verticalLayout_62.setSpacing(8)
        self.verticalLayout_62.setObjectName("verticalLayout_62")
        self.treeCsvButton_4 = QtWidgets.QRadioButton(self.widget_26)
        self.treeCsvButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_4.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_4.setSizePolicy(sizePolicy)
        self.treeCsvButton_4.setObjectName("treeCsvButton_4")
        self.verticalLayout_62.addWidget(self.treeCsvButton_4)
        self.treeCustomButton_4 = QtWidgets.QRadioButton(self.widget_26)
        self.treeCustomButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_4.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_4.setSizePolicy(sizePolicy)
        self.treeCustomButton_4.setObjectName("treeCustomButton_4")
        self.verticalLayout_62.addWidget(self.treeCustomButton_4)
        self.formLayout_33.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_62
        )
        self.treeCsvLabel_4 = QtWidgets.QLabel(self.widget_26)
        self.treeCsvLabel_4.setEnabled(False)
        self.treeCsvLabel_4.setObjectName("treeCsvLabel_4")
        self.formLayout_33.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_4
        )
        self.horizontalLayout_96 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_96.setSpacing(8)
        self.horizontalLayout_96.setObjectName("horizontalLayout_96")
        self.treeCsvBrowseBox_4 = QtWidgets.QLineEdit(self.widget_26)
        self.treeCsvBrowseBox_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_4.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_4.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_4.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_4.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_4.setMaxLength(1000)
        self.treeCsvBrowseBox_4.setObjectName("treeCsvBrowseBox_4")
        self.horizontalLayout_96.addWidget(self.treeCsvBrowseBox_4)
        self.treeCsvBrowseButton_4 = QtWidgets.QPushButton(self.widget_26)
        self.treeCsvBrowseButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_4.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_4.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_4.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_4.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_4.setObjectName("treeCsvBrowseButton_4")
        self.horizontalLayout_96.addWidget(self.treeCsvBrowseButton_4)
        self.formLayout_33.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_96
        )
        self.label_121 = QtWidgets.QLabel(self.widget_26)
        self.label_121.setObjectName("label_121")
        self.formLayout_33.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_121)
        self.horizontalLayout_97 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_97.setSpacing(8)
        self.horizontalLayout_97.setObjectName("horizontalLayout_97")
        self.standDens_4 = QtWidgets.QSpinBox(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_4.sizePolicy().hasHeightForWidth())
        self.standDens_4.setSizePolicy(sizePolicy)
        self.standDens_4.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_4.setMaximum(999)
        self.standDens_4.setSingleStep(50)
        self.standDens_4.setProperty("value", 200)
        self.standDens_4.setObjectName("standDens_4")
        self.horizontalLayout_97.addWidget(self.standDens_4)
        self.label_122 = QtWidgets.QLabel(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_122.sizePolicy().hasHeightForWidth())
        self.label_122.setSizePolicy(sizePolicy)
        self.label_122.setObjectName("label_122")
        self.horizontalLayout_97.addWidget(self.label_122)
        self.formLayout_33.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_97
        )
        self.label_25 = QtWidgets.QLabel(self.widget_26)
        self.label_25.setObjectName("label_25")
        self.formLayout_33.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.horizontalLayout_98 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_98.setSpacing(8)
        self.horizontalLayout_98.setObjectName("horizontalLayout_98")
        self.label_37 = QtWidgets.QLabel(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_37.sizePolicy().hasHeightForWidth())
        self.label_37.setSizePolicy(sizePolicy)
        self.label_37.setObjectName("label_37")
        self.horizontalLayout_98.addWidget(self.label_37)
        self.yearPlanted_4 = QtWidgets.QSpinBox(self.widget_26)
        self.yearPlanted_4.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_4.setObjectName("yearPlanted_4")
        self.horizontalLayout_98.addWidget(self.yearPlanted_4)
        self.formLayout_33.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_98
        )
        self.label_222 = QtWidgets.QLabel(self.widget_26)
        self.label_222.setObjectName("label_222")
        self.formLayout_33.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_222)
        self.speciesName_4 = QtWidgets.QLineEdit(self.widget_26)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_4.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_4.setSizePolicy(sizePolicy)
        self.speciesName_4.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_4.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_4.setObjectName("speciesName_4")
        self.formLayout_33.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_4
        )
        self.gridLayout_26.addLayout(self.formLayout_33, 1, 0, 1, 1)
        self.verticalLayout_63.addWidget(self.widget_26)
        self.stackedWidget.addWidget(self.treeMgmtPage_4)
        self.treeThinPage_4 = QtWidgets.QWidget()
        self.treeThinPage_4.setObjectName("treeThinPage_4")
        self.verticalLayout_66 = QtWidgets.QVBoxLayout(self.treeThinPage_4)
        self.verticalLayout_66.setObjectName("verticalLayout_66")
        self.widget_27 = QtWidgets.QWidget(self.treeThinPage_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_27.sizePolicy().hasHeightForWidth())
        self.widget_27.setSizePolicy(sizePolicy)
        self.widget_27.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_27.setObjectName("widget_27")
        self.gridLayout_27 = QtWidgets.QGridLayout(self.widget_27)
        self.gridLayout_27.setObjectName("gridLayout_27")
        self.label_123 = QtWidgets.QLabel(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_123.sizePolicy().hasHeightForWidth())
        self.label_123.setSizePolicy(sizePolicy)
        self.label_123.setMinimumSize(QtCore.QSize(0, 30))
        self.label_123.setObjectName("label_123")
        self.gridLayout_27.addWidget(self.label_123, 0, 0, 1, 1)
        self.formLayout_34 = QtWidgets.QFormLayout()
        self.formLayout_34.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_34.setVerticalSpacing(15)
        self.formLayout_34.setObjectName("formLayout_34")
        spacerItem31 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_34.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem31)
        self.label_40 = QtWidgets.QLabel(self.widget_27)
        self.label_40.setObjectName("label_40")
        self.formLayout_34.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_40)
        self.verticalLayout_64 = QtWidgets.QVBoxLayout()
        self.verticalLayout_64.setObjectName("verticalLayout_64")
        self.thinNever_4 = QtWidgets.QRadioButton(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_4.sizePolicy().hasHeightForWidth())
        self.thinNever_4.setSizePolicy(sizePolicy)
        self.thinNever_4.setObjectName("thinNever_4")
        self.verticalLayout_64.addWidget(self.thinNever_4)
        self.horizontalLayout_100 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_100.setSpacing(8)
        self.horizontalLayout_100.setObjectName("horizontalLayout_100")
        self.thinInterval_4 = QtWidgets.QRadioButton(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_4.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_4.setSizePolicy(sizePolicy)
        self.thinInterval_4.setObjectName("thinInterval_4")
        self.horizontalLayout_100.addWidget(self.thinInterval_4)
        self.thinFreq_4 = QtWidgets.QSpinBox(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_4.sizePolicy().hasHeightForWidth())
        self.thinFreq_4.setSizePolicy(sizePolicy)
        self.thinFreq_4.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_4.setMinimum(1)
        self.thinFreq_4.setSingleStep(10)
        self.thinFreq_4.setProperty("value", 1)
        self.thinFreq_4.setObjectName("thinFreq_4")
        self.horizontalLayout_100.addWidget(self.thinFreq_4)
        self.thinFreqLabel_4 = QtWidgets.QLabel(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_4.setSizePolicy(sizePolicy)
        self.thinFreqLabel_4.setObjectName("thinFreqLabel_4")
        self.horizontalLayout_100.addWidget(self.thinFreqLabel_4)
        self.verticalLayout_64.addLayout(self.horizontalLayout_100)
        self.thinCustom_4 = QtWidgets.QRadioButton(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_4.sizePolicy().hasHeightForWidth())
        self.thinCustom_4.setSizePolicy(sizePolicy)
        self.thinCustom_4.setObjectName("thinCustom_4")
        self.verticalLayout_64.addWidget(self.thinCustom_4)
        self.formLayout_34.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_64
        )
        self.thinCustomText_4 = QtWidgets.QLabel(self.widget_27)
        self.thinCustomText_4.setObjectName("thinCustomText_4")
        self.formLayout_34.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_4
        )
        self.thinCustomLine_4 = QtWidgets.QLineEdit(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_4.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_4.setSizePolicy(sizePolicy)
        self.thinCustomLine_4.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_4.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_4.setObjectName("thinCustomLine_4")
        self.formLayout_34.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_4
        )
        self.treesThinnedText_4 = QtWidgets.QLabel(self.widget_27)
        self.treesThinnedText_4.setObjectName("treesThinnedText_4")
        self.formLayout_34.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_4
        )
        self.horizontalLayout_101 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_101.setSpacing(8)
        self.horizontalLayout_101.setObjectName("horizontalLayout_101")
        self.treesThinned_4 = QtWidgets.QSpinBox(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_4.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_4.setSizePolicy(sizePolicy)
        self.treesThinned_4.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_4.setMaximum(100)
        self.treesThinned_4.setSingleStep(10)
        self.treesThinned_4.setProperty("value", 50)
        self.treesThinned_4.setObjectName("treesThinned_4")
        self.horizontalLayout_101.addWidget(self.treesThinned_4)
        self.treesThinnedLabel_4 = QtWidgets.QLabel(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_4.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_4.setObjectName("treesThinnedLabel_4")
        self.horizontalLayout_101.addWidget(self.treesThinnedLabel_4)
        self.formLayout_34.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_101
        )
        self.thinnedLeftText_4 = QtWidgets.QLabel(self.widget_27)
        self.thinnedLeftText_4.setObjectName("thinnedLeftText_4")
        self.formLayout_34.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_4
        )
        self.thinnedStemsLeftText_4 = QtWidgets.QLabel(self.widget_27)
        self.thinnedStemsLeftText_4.setObjectName("thinnedStemsLeftText_4")
        self.formLayout_34.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_4
        )
        self.horizontalLayout_102 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_102.setSpacing(8)
        self.horizontalLayout_102.setObjectName("horizontalLayout_102")
        self.thinnedStemsLeft_4 = QtWidgets.QSpinBox(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_4.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_4.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_4.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_4.setMaximum(100)
        self.thinnedStemsLeft_4.setSingleStep(10)
        self.thinnedStemsLeft_4.setObjectName("thinnedStemsLeft_4")
        self.horizontalLayout_102.addWidget(self.thinnedStemsLeft_4)
        self.thinnedStemsLeftLabel_4 = QtWidgets.QLabel(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_4.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_4.setObjectName("thinnedStemsLeftLabel_4")
        self.horizontalLayout_102.addWidget(self.thinnedStemsLeftLabel_4)
        self.formLayout_34.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_102
        )
        self.thinnedBranchesLeftText_4 = QtWidgets.QLabel(self.widget_27)
        self.thinnedBranchesLeftText_4.setObjectName("thinnedBranchesLeftText_4")
        self.formLayout_34.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_4
        )
        self.horizontalLayout_103 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_103.setSpacing(8)
        self.horizontalLayout_103.setObjectName("horizontalLayout_103")
        self.thinnedBranchesLeft_4 = QtWidgets.QSpinBox(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_4.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_4.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_4.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_4.setMaximum(100)
        self.thinnedBranchesLeft_4.setSingleStep(10)
        self.thinnedBranchesLeft_4.setObjectName("thinnedBranchesLeft_4")
        self.horizontalLayout_103.addWidget(self.thinnedBranchesLeft_4)
        self.thinnedBranchesLeftLabel_4 = QtWidgets.QLabel(self.widget_27)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_4.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_4.setObjectName("thinnedBranchesLeftLabel_4")
        self.horizontalLayout_103.addWidget(self.thinnedBranchesLeftLabel_4)
        self.formLayout_34.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_103
        )
        self.gridLayout_27.addLayout(self.formLayout_34, 1, 0, 1, 1)
        self.verticalLayout_66.addWidget(self.widget_27)
        self.stackedWidget.addWidget(self.treeThinPage_4)
        self.treeMortPage_4 = QtWidgets.QWidget()
        self.treeMortPage_4.setObjectName("treeMortPage_4")
        self.verticalLayout_67 = QtWidgets.QVBoxLayout(self.treeMortPage_4)
        self.verticalLayout_67.setObjectName("verticalLayout_67")
        self.widget_28 = QtWidgets.QWidget(self.treeMortPage_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_28.sizePolicy().hasHeightForWidth())
        self.widget_28.setSizePolicy(sizePolicy)
        self.widget_28.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_28.setObjectName("widget_28")
        self.gridLayout_28 = QtWidgets.QGridLayout(self.widget_28)
        self.gridLayout_28.setObjectName("gridLayout_28")
        self.label_124 = QtWidgets.QLabel(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_124.sizePolicy().hasHeightForWidth())
        self.label_124.setSizePolicy(sizePolicy)
        self.label_124.setMinimumSize(QtCore.QSize(0, 30))
        self.label_124.setObjectName("label_124")
        self.gridLayout_28.addWidget(self.label_124, 0, 0, 1, 1)
        self.formLayout_35 = QtWidgets.QFormLayout()
        self.formLayout_35.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_35.setVerticalSpacing(15)
        self.formLayout_35.setObjectName("formLayout_35")
        spacerItem32 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_35.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem32)
        self.label_125 = QtWidgets.QLabel(self.widget_28)
        self.label_125.setObjectName("label_125")
        self.formLayout_35.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_125)
        self.horizontalLayout_104 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_104.setSpacing(8)
        self.horizontalLayout_104.setObjectName("horizontalLayout_104")
        self.treesDead_4 = QtWidgets.QSpinBox(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_4.sizePolicy().hasHeightForWidth())
        self.treesDead_4.setSizePolicy(sizePolicy)
        self.treesDead_4.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_4.setMaximum(100)
        self.treesDead_4.setSingleStep(10)
        self.treesDead_4.setProperty("value", 1)
        self.treesDead_4.setObjectName("treesDead_4")
        self.horizontalLayout_104.addWidget(self.treesDead_4)
        self.label_126 = QtWidgets.QLabel(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_126.sizePolicy().hasHeightForWidth())
        self.label_126.setSizePolicy(sizePolicy)
        self.label_126.setObjectName("label_126")
        self.horizontalLayout_104.addWidget(self.label_126)
        self.formLayout_35.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_104
        )
        self.deadLeftText_4 = QtWidgets.QLabel(self.widget_28)
        self.deadLeftText_4.setObjectName("deadLeftText_4")
        self.formLayout_35.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_4
        )
        self.deadStemsLeftText_4 = QtWidgets.QLabel(self.widget_28)
        self.deadStemsLeftText_4.setObjectName("deadStemsLeftText_4")
        self.formLayout_35.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_4
        )
        self.horizontalLayout_105 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_105.setSpacing(8)
        self.horizontalLayout_105.setObjectName("horizontalLayout_105")
        self.deadStemsLeft_4 = QtWidgets.QSpinBox(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_4.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_4.setSizePolicy(sizePolicy)
        self.deadStemsLeft_4.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_4.setMaximum(100)
        self.deadStemsLeft_4.setSingleStep(10)
        self.deadStemsLeft_4.setObjectName("deadStemsLeft_4")
        self.horizontalLayout_105.addWidget(self.deadStemsLeft_4)
        self.deadStemsLeftLabel_4 = QtWidgets.QLabel(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_4.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_4.setObjectName("deadStemsLeftLabel_4")
        self.horizontalLayout_105.addWidget(self.deadStemsLeftLabel_4)
        self.formLayout_35.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_105
        )
        self.deadBranchesLeftText_4 = QtWidgets.QLabel(self.widget_28)
        self.deadBranchesLeftText_4.setObjectName("deadBranchesLeftText_4")
        self.formLayout_35.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_4
        )
        self.horizontalLayout_106 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_106.setSpacing(8)
        self.horizontalLayout_106.setObjectName("horizontalLayout_106")
        self.deadBranchesLeft_4 = QtWidgets.QSpinBox(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_4.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_4.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_4.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_4.setMaximum(100)
        self.deadBranchesLeft_4.setSingleStep(10)
        self.deadBranchesLeft_4.setObjectName("deadBranchesLeft_4")
        self.horizontalLayout_106.addWidget(self.deadBranchesLeft_4)
        self.deadBranchesLeftLabel_4 = QtWidgets.QLabel(self.widget_28)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_4.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_4.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_4.setObjectName("deadBranchesLeftLabel_4")
        self.horizontalLayout_106.addWidget(self.deadBranchesLeftLabel_4)
        self.formLayout_35.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_106
        )
        self.gridLayout_28.addLayout(self.formLayout_35, 1, 0, 1, 1)
        self.verticalLayout_67.addWidget(self.widget_28)
        self.stackedWidget.addWidget(self.treeMortPage_4)
        self.treeGrowthPage_4 = QtWidgets.QWidget()
        self.treeGrowthPage_4.setObjectName("treeGrowthPage_4")
        self.verticalLayout_69 = QtWidgets.QVBoxLayout(self.treeGrowthPage_4)
        self.verticalLayout_69.setObjectName("verticalLayout_69")
        self.widget_29 = QtWidgets.QWidget(self.treeGrowthPage_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_29.sizePolicy().hasHeightForWidth())
        self.widget_29.setSizePolicy(sizePolicy)
        self.widget_29.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_29.setObjectName("widget_29")
        self.gridLayout_29 = QtWidgets.QGridLayout(self.widget_29)
        self.gridLayout_29.setObjectName("gridLayout_29")
        self.formLayout_36 = QtWidgets.QFormLayout()
        self.formLayout_36.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_36.setVerticalSpacing(15)
        self.formLayout_36.setObjectName("formLayout_36")
        spacerItem33 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_36.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem33)
        self.verticalLayout_68 = QtWidgets.QVBoxLayout()
        self.verticalLayout_68.setSpacing(8)
        self.verticalLayout_68.setObjectName("verticalLayout_68")
        self.growthCsvButton_4 = QtWidgets.QRadioButton(self.widget_29)
        self.growthCsvButton_4.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_4.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_4.setSizePolicy(sizePolicy)
        self.growthCsvButton_4.setObjectName("growthCsvButton_4")
        self.verticalLayout_68.addWidget(self.growthCsvButton_4)
        self.growthCustomButton_4 = QtWidgets.QRadioButton(self.widget_29)
        self.growthCustomButton_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_4.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_4.setSizePolicy(sizePolicy)
        self.growthCustomButton_4.setObjectName("growthCustomButton_4")
        self.verticalLayout_68.addWidget(self.growthCustomButton_4)
        self.formLayout_36.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_68
        )
        self.growthCsvLabel_4 = QtWidgets.QLabel(self.widget_29)
        self.growthCsvLabel_4.setEnabled(True)
        self.growthCsvLabel_4.setObjectName("growthCsvLabel_4")
        self.formLayout_36.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_4
        )
        self.horizontalLayout_107 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_107.setSpacing(8)
        self.horizontalLayout_107.setObjectName("horizontalLayout_107")
        self.growthCsvBrowseBox_4 = QtWidgets.QLineEdit(self.widget_29)
        self.growthCsvBrowseBox_4.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_4.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_4.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_4.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_4.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_4.setMaxLength(1000)
        self.growthCsvBrowseBox_4.setObjectName("growthCsvBrowseBox_4")
        self.horizontalLayout_107.addWidget(self.growthCsvBrowseBox_4)
        self.growthCsvBrowseButton_4 = QtWidgets.QPushButton(self.widget_29)
        self.growthCsvBrowseButton_4.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_4.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_4.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_4.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_4.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_4.setObjectName("growthCsvBrowseButton_4")
        self.horizontalLayout_107.addWidget(self.growthCsvBrowseButton_4)
        self.formLayout_36.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_107
        )
        self.label_127 = QtWidgets.QLabel(self.widget_29)
        self.label_127.setObjectName("label_127")
        self.formLayout_36.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_127)
        self.allom_4 = QtWidgets.QComboBox(self.widget_29)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_4.sizePolicy().hasHeightForWidth())
        self.allom_4.setSizePolicy(sizePolicy)
        self.allom_4.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_4.setObjectName("allom_4")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.setItemText(1, "")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.setItemText(7, "")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.allom_4.setItemText(12, "")
        self.allom_4.addItem("")
        self.allom_4.addItem("")
        self.formLayout_36.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_4)
        self.gridLayout_29.addLayout(self.formLayout_36, 1, 0, 1, 1)
        self.label_128 = QtWidgets.QLabel(self.widget_29)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_128.sizePolicy().hasHeightForWidth())
        self.label_128.setSizePolicy(sizePolicy)
        self.label_128.setMinimumSize(QtCore.QSize(0, 30))
        self.label_128.setObjectName("label_128")
        self.gridLayout_29.addWidget(self.label_128, 0, 0, 1, 1)
        self.verticalLayout_69.addWidget(self.widget_29)
        self.stackedWidget.addWidget(self.treeGrowthPage_4)
        self.treeMgmtPage_5 = QtWidgets.QWidget()
        self.treeMgmtPage_5.setObjectName("treeMgmtPage_5")
        self.verticalLayout_81 = QtWidgets.QVBoxLayout(self.treeMgmtPage_5)
        self.verticalLayout_81.setObjectName("verticalLayout_81")
        self.widget_30 = QtWidgets.QWidget(self.treeMgmtPage_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_30.sizePolicy().hasHeightForWidth())
        self.widget_30.setSizePolicy(sizePolicy)
        self.widget_30.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_30.setObjectName("widget_30")
        self.gridLayout_35 = QtWidgets.QGridLayout(self.widget_30)
        self.gridLayout_35.setObjectName("gridLayout_35")
        self.label_159 = QtWidgets.QLabel(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_159.sizePolicy().hasHeightForWidth())
        self.label_159.setSizePolicy(sizePolicy)
        self.label_159.setMinimumSize(QtCore.QSize(0, 30))
        self.label_159.setObjectName("label_159")
        self.gridLayout_35.addWidget(self.label_159, 0, 0, 1, 1)
        self.formLayout_42 = QtWidgets.QFormLayout()
        self.formLayout_42.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_42.setVerticalSpacing(15)
        self.formLayout_42.setObjectName("formLayout_42")
        spacerItem34 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_42.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem34)
        self.label_160 = QtWidgets.QLabel(self.widget_30)
        self.label_160.setObjectName("label_160")
        self.formLayout_42.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_160)
        self.treeType_5 = QtWidgets.QComboBox(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_5.sizePolicy().hasHeightForWidth())
        self.treeType_5.setSizePolicy(sizePolicy)
        self.treeType_5.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_5.setObjectName("treeType_5")
        self.treeType_5.addItem("")
        self.treeType_5.addItem("")
        self.treeType_5.setItemText(1, "")
        self.treeType_5.addItem("")
        self.treeType_5.addItem("")
        self.treeType_5.addItem("")
        self.treeType_5.setItemText(4, "")
        self.treeType_5.addItem("")
        self.treeType_5.addItem("")
        self.formLayout_42.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_5
        )
        self.treeInputType_5 = QtWidgets.QLabel(self.widget_30)
        self.treeInputType_5.setEnabled(False)
        self.treeInputType_5.setObjectName("treeInputType_5")
        self.formLayout_42.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_5
        )
        self.verticalLayout_80 = QtWidgets.QVBoxLayout()
        self.verticalLayout_80.setSpacing(8)
        self.verticalLayout_80.setObjectName("verticalLayout_80")
        self.treeCsvButton_5 = QtWidgets.QRadioButton(self.widget_30)
        self.treeCsvButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_5.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_5.setSizePolicy(sizePolicy)
        self.treeCsvButton_5.setObjectName("treeCsvButton_5")
        self.verticalLayout_80.addWidget(self.treeCsvButton_5)
        self.treeCustomButton_5 = QtWidgets.QRadioButton(self.widget_30)
        self.treeCustomButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_5.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_5.setSizePolicy(sizePolicy)
        self.treeCustomButton_5.setObjectName("treeCustomButton_5")
        self.verticalLayout_80.addWidget(self.treeCustomButton_5)
        self.formLayout_42.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_80
        )
        self.treeCsvLabel_5 = QtWidgets.QLabel(self.widget_30)
        self.treeCsvLabel_5.setEnabled(False)
        self.treeCsvLabel_5.setObjectName("treeCsvLabel_5")
        self.formLayout_42.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_5
        )
        self.horizontalLayout_124 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_124.setSpacing(8)
        self.horizontalLayout_124.setObjectName("horizontalLayout_124")
        self.treeCsvBrowseBox_5 = QtWidgets.QLineEdit(self.widget_30)
        self.treeCsvBrowseBox_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_5.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_5.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_5.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_5.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_5.setMaxLength(1000)
        self.treeCsvBrowseBox_5.setObjectName("treeCsvBrowseBox_5")
        self.horizontalLayout_124.addWidget(self.treeCsvBrowseBox_5)
        self.treeCsvBrowseButton_5 = QtWidgets.QPushButton(self.widget_30)
        self.treeCsvBrowseButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_5.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_5.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_5.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_5.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_5.setObjectName("treeCsvBrowseButton_5")
        self.horizontalLayout_124.addWidget(self.treeCsvBrowseButton_5)
        self.formLayout_42.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_124
        )
        self.label_161 = QtWidgets.QLabel(self.widget_30)
        self.label_161.setObjectName("label_161")
        self.formLayout_42.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_161)
        self.horizontalLayout_125 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_125.setSpacing(8)
        self.horizontalLayout_125.setObjectName("horizontalLayout_125")
        self.standDens_5 = QtWidgets.QSpinBox(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_5.sizePolicy().hasHeightForWidth())
        self.standDens_5.setSizePolicy(sizePolicy)
        self.standDens_5.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_5.setMaximum(999)
        self.standDens_5.setSingleStep(50)
        self.standDens_5.setProperty("value", 200)
        self.standDens_5.setObjectName("standDens_5")
        self.horizontalLayout_125.addWidget(self.standDens_5)
        self.label_162 = QtWidgets.QLabel(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_162.sizePolicy().hasHeightForWidth())
        self.label_162.setSizePolicy(sizePolicy)
        self.label_162.setObjectName("label_162")
        self.horizontalLayout_125.addWidget(self.label_162)
        self.formLayout_42.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_125
        )
        self.label_42 = QtWidgets.QLabel(self.widget_30)
        self.label_42.setObjectName("label_42")
        self.formLayout_42.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_42)
        self.horizontalLayout_126 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_126.setSpacing(8)
        self.horizontalLayout_126.setObjectName("horizontalLayout_126")
        self.label_43 = QtWidgets.QLabel(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_43.sizePolicy().hasHeightForWidth())
        self.label_43.setSizePolicy(sizePolicy)
        self.label_43.setObjectName("label_43")
        self.horizontalLayout_126.addWidget(self.label_43)
        self.yearPlanted_5 = QtWidgets.QSpinBox(self.widget_30)
        self.yearPlanted_5.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_5.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_5.setObjectName("yearPlanted_5")
        self.horizontalLayout_126.addWidget(self.yearPlanted_5)
        self.formLayout_42.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_126
        )
        self.label_62 = QtWidgets.QLabel(self.widget_30)
        self.label_62.setObjectName("label_62")
        self.formLayout_42.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_62)
        self.speciesName_5 = QtWidgets.QLineEdit(self.widget_30)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_5.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_5.setSizePolicy(sizePolicy)
        self.speciesName_5.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_5.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_5.setObjectName("speciesName_5")
        self.formLayout_42.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_5
        )
        self.gridLayout_35.addLayout(self.formLayout_42, 1, 0, 1, 1)
        self.verticalLayout_81.addWidget(self.widget_30)
        self.stackedWidget.addWidget(self.treeMgmtPage_5)
        self.treeThinPage_5 = QtWidgets.QWidget()
        self.treeThinPage_5.setObjectName("treeThinPage_5")
        self.verticalLayout_83 = QtWidgets.QVBoxLayout(self.treeThinPage_5)
        self.verticalLayout_83.setObjectName("verticalLayout_83")
        self.widget_31 = QtWidgets.QWidget(self.treeThinPage_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_31.sizePolicy().hasHeightForWidth())
        self.widget_31.setSizePolicy(sizePolicy)
        self.widget_31.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_31.setObjectName("widget_31")
        self.gridLayout_36 = QtWidgets.QGridLayout(self.widget_31)
        self.gridLayout_36.setObjectName("gridLayout_36")
        self.label_163 = QtWidgets.QLabel(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_163.sizePolicy().hasHeightForWidth())
        self.label_163.setSizePolicy(sizePolicy)
        self.label_163.setMinimumSize(QtCore.QSize(0, 30))
        self.label_163.setObjectName("label_163")
        self.gridLayout_36.addWidget(self.label_163, 0, 0, 1, 1)
        self.formLayout_43 = QtWidgets.QFormLayout()
        self.formLayout_43.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_43.setVerticalSpacing(15)
        self.formLayout_43.setObjectName("formLayout_43")
        spacerItem35 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_43.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem35)
        self.label_164 = QtWidgets.QLabel(self.widget_31)
        self.label_164.setObjectName("label_164")
        self.formLayout_43.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_164)
        self.verticalLayout_82 = QtWidgets.QVBoxLayout()
        self.verticalLayout_82.setObjectName("verticalLayout_82")
        self.thinNever_5 = QtWidgets.QRadioButton(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_5.sizePolicy().hasHeightForWidth())
        self.thinNever_5.setSizePolicy(sizePolicy)
        self.thinNever_5.setObjectName("thinNever_5")
        self.verticalLayout_82.addWidget(self.thinNever_5)
        self.horizontalLayout_127 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_127.setSpacing(8)
        self.horizontalLayout_127.setObjectName("horizontalLayout_127")
        self.thinInterval_5 = QtWidgets.QRadioButton(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_5.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_5.setSizePolicy(sizePolicy)
        self.thinInterval_5.setObjectName("thinInterval_5")
        self.horizontalLayout_127.addWidget(self.thinInterval_5)
        self.thinFreq_5 = QtWidgets.QSpinBox(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_5.sizePolicy().hasHeightForWidth())
        self.thinFreq_5.setSizePolicy(sizePolicy)
        self.thinFreq_5.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_5.setMinimum(1)
        self.thinFreq_5.setSingleStep(10)
        self.thinFreq_5.setProperty("value", 1)
        self.thinFreq_5.setObjectName("thinFreq_5")
        self.horizontalLayout_127.addWidget(self.thinFreq_5)
        self.thinFreqLabel_5 = QtWidgets.QLabel(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_5.setSizePolicy(sizePolicy)
        self.thinFreqLabel_5.setObjectName("thinFreqLabel_5")
        self.horizontalLayout_127.addWidget(self.thinFreqLabel_5)
        self.verticalLayout_82.addLayout(self.horizontalLayout_127)
        self.thinCustom_5 = QtWidgets.QRadioButton(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_5.sizePolicy().hasHeightForWidth())
        self.thinCustom_5.setSizePolicy(sizePolicy)
        self.thinCustom_5.setObjectName("thinCustom_5")
        self.verticalLayout_82.addWidget(self.thinCustom_5)
        self.formLayout_43.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_82
        )
        self.thinCustomText_5 = QtWidgets.QLabel(self.widget_31)
        self.thinCustomText_5.setObjectName("thinCustomText_5")
        self.formLayout_43.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_5
        )
        self.thinCustomLine_5 = QtWidgets.QLineEdit(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_5.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_5.setSizePolicy(sizePolicy)
        self.thinCustomLine_5.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_5.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_5.setObjectName("thinCustomLine_5")
        self.formLayout_43.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_5
        )
        self.treesThinnedText_5 = QtWidgets.QLabel(self.widget_31)
        self.treesThinnedText_5.setObjectName("treesThinnedText_5")
        self.formLayout_43.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_5
        )
        self.horizontalLayout_128 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_128.setSpacing(8)
        self.horizontalLayout_128.setObjectName("horizontalLayout_128")
        self.treesThinned_5 = QtWidgets.QSpinBox(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_5.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_5.setSizePolicy(sizePolicy)
        self.treesThinned_5.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_5.setMaximum(100)
        self.treesThinned_5.setSingleStep(10)
        self.treesThinned_5.setProperty("value", 50)
        self.treesThinned_5.setObjectName("treesThinned_5")
        self.horizontalLayout_128.addWidget(self.treesThinned_5)
        self.treesThinnedLabel_5 = QtWidgets.QLabel(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_5.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_5.setObjectName("treesThinnedLabel_5")
        self.horizontalLayout_128.addWidget(self.treesThinnedLabel_5)
        self.formLayout_43.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_128
        )
        self.thinnedLeftText_5 = QtWidgets.QLabel(self.widget_31)
        self.thinnedLeftText_5.setObjectName("thinnedLeftText_5")
        self.formLayout_43.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_5
        )
        self.thinnedStemsLeftText_5 = QtWidgets.QLabel(self.widget_31)
        self.thinnedStemsLeftText_5.setObjectName("thinnedStemsLeftText_5")
        self.formLayout_43.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_5
        )
        self.horizontalLayout_129 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_129.setSpacing(8)
        self.horizontalLayout_129.setObjectName("horizontalLayout_129")
        self.thinnedStemsLeft_5 = QtWidgets.QSpinBox(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_5.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_5.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_5.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_5.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_5.setMaximum(100)
        self.thinnedStemsLeft_5.setSingleStep(10)
        self.thinnedStemsLeft_5.setObjectName("thinnedStemsLeft_5")
        self.horizontalLayout_129.addWidget(self.thinnedStemsLeft_5)
        self.thinnedStemsLeftLabel_5 = QtWidgets.QLabel(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_5.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_5.setObjectName("thinnedStemsLeftLabel_5")
        self.horizontalLayout_129.addWidget(self.thinnedStemsLeftLabel_5)
        self.formLayout_43.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_129
        )
        self.thinnedBranchesLeftText_5 = QtWidgets.QLabel(self.widget_31)
        self.thinnedBranchesLeftText_5.setObjectName("thinnedBranchesLeftText_5")
        self.formLayout_43.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_5
        )
        self.horizontalLayout_130 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_130.setSpacing(8)
        self.horizontalLayout_130.setObjectName("horizontalLayout_130")
        self.thinnedBranchesLeft_5 = QtWidgets.QSpinBox(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_5.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_5.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_5.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_5.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_5.setMaximum(100)
        self.thinnedBranchesLeft_5.setSingleStep(10)
        self.thinnedBranchesLeft_5.setObjectName("thinnedBranchesLeft_5")
        self.horizontalLayout_130.addWidget(self.thinnedBranchesLeft_5)
        self.thinnedBranchesLeftLabel_5 = QtWidgets.QLabel(self.widget_31)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_5.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_5.setObjectName("thinnedBranchesLeftLabel_5")
        self.horizontalLayout_130.addWidget(self.thinnedBranchesLeftLabel_5)
        self.formLayout_43.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_130
        )
        self.gridLayout_36.addLayout(self.formLayout_43, 1, 0, 1, 1)
        self.verticalLayout_83.addWidget(self.widget_31)
        self.stackedWidget.addWidget(self.treeThinPage_5)
        self.treeMortPage_5 = QtWidgets.QWidget()
        self.treeMortPage_5.setObjectName("treeMortPage_5")
        self.verticalLayout_84 = QtWidgets.QVBoxLayout(self.treeMortPage_5)
        self.verticalLayout_84.setObjectName("verticalLayout_84")
        self.widget_32 = QtWidgets.QWidget(self.treeMortPage_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_32.sizePolicy().hasHeightForWidth())
        self.widget_32.setSizePolicy(sizePolicy)
        self.widget_32.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_32.setObjectName("widget_32")
        self.gridLayout_37 = QtWidgets.QGridLayout(self.widget_32)
        self.gridLayout_37.setObjectName("gridLayout_37")
        self.label_165 = QtWidgets.QLabel(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_165.sizePolicy().hasHeightForWidth())
        self.label_165.setSizePolicy(sizePolicy)
        self.label_165.setMinimumSize(QtCore.QSize(0, 30))
        self.label_165.setObjectName("label_165")
        self.gridLayout_37.addWidget(self.label_165, 0, 0, 1, 1)
        self.formLayout_44 = QtWidgets.QFormLayout()
        self.formLayout_44.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_44.setVerticalSpacing(15)
        self.formLayout_44.setObjectName("formLayout_44")
        spacerItem36 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_44.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem36)
        self.label_166 = QtWidgets.QLabel(self.widget_32)
        self.label_166.setObjectName("label_166")
        self.formLayout_44.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_166)
        self.horizontalLayout_131 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_131.setSpacing(8)
        self.horizontalLayout_131.setObjectName("horizontalLayout_131")
        self.treesDead_5 = QtWidgets.QSpinBox(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_5.sizePolicy().hasHeightForWidth())
        self.treesDead_5.setSizePolicy(sizePolicy)
        self.treesDead_5.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_5.setMaximum(100)
        self.treesDead_5.setSingleStep(10)
        self.treesDead_5.setProperty("value", 1)
        self.treesDead_5.setObjectName("treesDead_5")
        self.horizontalLayout_131.addWidget(self.treesDead_5)
        self.label_167 = QtWidgets.QLabel(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_167.sizePolicy().hasHeightForWidth())
        self.label_167.setSizePolicy(sizePolicy)
        self.label_167.setObjectName("label_167")
        self.horizontalLayout_131.addWidget(self.label_167)
        self.formLayout_44.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_131
        )
        self.deadLeftText_5 = QtWidgets.QLabel(self.widget_32)
        self.deadLeftText_5.setObjectName("deadLeftText_5")
        self.formLayout_44.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_5
        )
        self.deadStemsLeftText_5 = QtWidgets.QLabel(self.widget_32)
        self.deadStemsLeftText_5.setObjectName("deadStemsLeftText_5")
        self.formLayout_44.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_5
        )
        self.horizontalLayout_132 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_132.setSpacing(8)
        self.horizontalLayout_132.setObjectName("horizontalLayout_132")
        self.deadStemsLeft_5 = QtWidgets.QSpinBox(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_5.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_5.setSizePolicy(sizePolicy)
        self.deadStemsLeft_5.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_5.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_5.setMaximum(100)
        self.deadStemsLeft_5.setSingleStep(10)
        self.deadStemsLeft_5.setObjectName("deadStemsLeft_5")
        self.horizontalLayout_132.addWidget(self.deadStemsLeft_5)
        self.deadStemsLeftLabel_5 = QtWidgets.QLabel(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_5.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_5.setObjectName("deadStemsLeftLabel_5")
        self.horizontalLayout_132.addWidget(self.deadStemsLeftLabel_5)
        self.formLayout_44.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_132
        )
        self.deadBranchesLeftText_5 = QtWidgets.QLabel(self.widget_32)
        self.deadBranchesLeftText_5.setObjectName("deadBranchesLeftText_5")
        self.formLayout_44.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_5
        )
        self.horizontalLayout_133 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_133.setSpacing(8)
        self.horizontalLayout_133.setObjectName("horizontalLayout_133")
        self.deadBranchesLeft_5 = QtWidgets.QSpinBox(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_5.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_5.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_5.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_5.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_5.setMaximum(100)
        self.deadBranchesLeft_5.setSingleStep(10)
        self.deadBranchesLeft_5.setObjectName("deadBranchesLeft_5")
        self.horizontalLayout_133.addWidget(self.deadBranchesLeft_5)
        self.deadBranchesLeftLabel_5 = QtWidgets.QLabel(self.widget_32)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_5.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_5.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_5.setObjectName("deadBranchesLeftLabel_5")
        self.horizontalLayout_133.addWidget(self.deadBranchesLeftLabel_5)
        self.formLayout_44.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_133
        )
        self.gridLayout_37.addLayout(self.formLayout_44, 1, 0, 1, 1)
        self.verticalLayout_84.addWidget(self.widget_32)
        self.stackedWidget.addWidget(self.treeMortPage_5)
        self.treeGrowthPage_5 = QtWidgets.QWidget()
        self.treeGrowthPage_5.setObjectName("treeGrowthPage_5")
        self.verticalLayout_86 = QtWidgets.QVBoxLayout(self.treeGrowthPage_5)
        self.verticalLayout_86.setObjectName("verticalLayout_86")
        self.widget_33 = QtWidgets.QWidget(self.treeGrowthPage_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_33.sizePolicy().hasHeightForWidth())
        self.widget_33.setSizePolicy(sizePolicy)
        self.widget_33.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_33.setObjectName("widget_33")
        self.gridLayout_38 = QtWidgets.QGridLayout(self.widget_33)
        self.gridLayout_38.setObjectName("gridLayout_38")
        self.formLayout_45 = QtWidgets.QFormLayout()
        self.formLayout_45.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_45.setVerticalSpacing(15)
        self.formLayout_45.setObjectName("formLayout_45")
        spacerItem37 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_45.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem37)
        self.verticalLayout_85 = QtWidgets.QVBoxLayout()
        self.verticalLayout_85.setSpacing(8)
        self.verticalLayout_85.setObjectName("verticalLayout_85")
        self.growthCsvButton_5 = QtWidgets.QRadioButton(self.widget_33)
        self.growthCsvButton_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_5.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_5.setSizePolicy(sizePolicy)
        self.growthCsvButton_5.setObjectName("growthCsvButton_5")
        self.verticalLayout_85.addWidget(self.growthCsvButton_5)
        self.growthCustomButton_5 = QtWidgets.QRadioButton(self.widget_33)
        self.growthCustomButton_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_5.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_5.setSizePolicy(sizePolicy)
        self.growthCustomButton_5.setObjectName("growthCustomButton_5")
        self.verticalLayout_85.addWidget(self.growthCustomButton_5)
        self.formLayout_45.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_85
        )
        self.growthCsvLabel_5 = QtWidgets.QLabel(self.widget_33)
        self.growthCsvLabel_5.setEnabled(True)
        self.growthCsvLabel_5.setObjectName("growthCsvLabel_5")
        self.formLayout_45.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_5
        )
        self.horizontalLayout_134 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_134.setSpacing(8)
        self.horizontalLayout_134.setObjectName("horizontalLayout_134")
        self.growthCsvBrowseBox_5 = QtWidgets.QLineEdit(self.widget_33)
        self.growthCsvBrowseBox_5.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_5.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_5.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_5.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_5.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_5.setMaxLength(1000)
        self.growthCsvBrowseBox_5.setObjectName("growthCsvBrowseBox_5")
        self.horizontalLayout_134.addWidget(self.growthCsvBrowseBox_5)
        self.growthCsvBrowseButton_5 = QtWidgets.QPushButton(self.widget_33)
        self.growthCsvBrowseButton_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_5.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_5.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_5.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_5.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_5.setObjectName("growthCsvBrowseButton_5")
        self.horizontalLayout_134.addWidget(self.growthCsvBrowseButton_5)
        self.formLayout_45.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_134
        )
        self.label_168 = QtWidgets.QLabel(self.widget_33)
        self.label_168.setObjectName("label_168")
        self.formLayout_45.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_168)
        self.allom_5 = QtWidgets.QComboBox(self.widget_33)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_5.sizePolicy().hasHeightForWidth())
        self.allom_5.setSizePolicy(sizePolicy)
        self.allom_5.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_5.setObjectName("allom_5")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.setItemText(1, "")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.setItemText(7, "")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.allom_5.setItemText(12, "")
        self.allom_5.addItem("")
        self.allom_5.addItem("")
        self.formLayout_45.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_5)
        self.gridLayout_38.addLayout(self.formLayout_45, 1, 0, 1, 1)
        self.label_169 = QtWidgets.QLabel(self.widget_33)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_169.sizePolicy().hasHeightForWidth())
        self.label_169.setSizePolicy(sizePolicy)
        self.label_169.setMinimumSize(QtCore.QSize(0, 30))
        self.label_169.setObjectName("label_169")
        self.gridLayout_38.addWidget(self.label_169, 0, 0, 1, 1)
        self.verticalLayout_86.addWidget(self.widget_33)
        self.stackedWidget.addWidget(self.treeGrowthPage_5)
        self.treeMgmtPage_6 = QtWidgets.QWidget()
        self.treeMgmtPage_6.setObjectName("treeMgmtPage_6")
        self.verticalLayout_88 = QtWidgets.QVBoxLayout(self.treeMgmtPage_6)
        self.verticalLayout_88.setObjectName("verticalLayout_88")
        self.widget_34 = QtWidgets.QWidget(self.treeMgmtPage_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_34.sizePolicy().hasHeightForWidth())
        self.widget_34.setSizePolicy(sizePolicy)
        self.widget_34.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_34.setObjectName("widget_34")
        self.gridLayout_39 = QtWidgets.QGridLayout(self.widget_34)
        self.gridLayout_39.setObjectName("gridLayout_39")
        self.label_170 = QtWidgets.QLabel(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_170.sizePolicy().hasHeightForWidth())
        self.label_170.setSizePolicy(sizePolicy)
        self.label_170.setMinimumSize(QtCore.QSize(0, 30))
        self.label_170.setObjectName("label_170")
        self.gridLayout_39.addWidget(self.label_170, 0, 0, 1, 1)
        self.formLayout_46 = QtWidgets.QFormLayout()
        self.formLayout_46.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_46.setVerticalSpacing(15)
        self.formLayout_46.setObjectName("formLayout_46")
        spacerItem38 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_46.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem38)
        self.label_171 = QtWidgets.QLabel(self.widget_34)
        self.label_171.setObjectName("label_171")
        self.formLayout_46.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_171)
        self.treeType_6 = QtWidgets.QComboBox(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_6.sizePolicy().hasHeightForWidth())
        self.treeType_6.setSizePolicy(sizePolicy)
        self.treeType_6.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_6.setObjectName("treeType_6")
        self.treeType_6.addItem("")
        self.treeType_6.addItem("")
        self.treeType_6.setItemText(1, "")
        self.treeType_6.addItem("")
        self.treeType_6.addItem("")
        self.treeType_6.addItem("")
        self.treeType_6.setItemText(4, "")
        self.treeType_6.addItem("")
        self.treeType_6.addItem("")
        self.formLayout_46.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_6
        )
        self.treeInputType_6 = QtWidgets.QLabel(self.widget_34)
        self.treeInputType_6.setEnabled(False)
        self.treeInputType_6.setObjectName("treeInputType_6")
        self.formLayout_46.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_6
        )
        self.verticalLayout_87 = QtWidgets.QVBoxLayout()
        self.verticalLayout_87.setSpacing(8)
        self.verticalLayout_87.setObjectName("verticalLayout_87")
        self.treeCsvButton_6 = QtWidgets.QRadioButton(self.widget_34)
        self.treeCsvButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_6.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_6.setSizePolicy(sizePolicy)
        self.treeCsvButton_6.setObjectName("treeCsvButton_6")
        self.verticalLayout_87.addWidget(self.treeCsvButton_6)
        self.treeCustomButton_6 = QtWidgets.QRadioButton(self.widget_34)
        self.treeCustomButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_6.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_6.setSizePolicy(sizePolicy)
        self.treeCustomButton_6.setObjectName("treeCustomButton_6")
        self.verticalLayout_87.addWidget(self.treeCustomButton_6)
        self.formLayout_46.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_87
        )
        self.treeCsvLabel_6 = QtWidgets.QLabel(self.widget_34)
        self.treeCsvLabel_6.setEnabled(False)
        self.treeCsvLabel_6.setObjectName("treeCsvLabel_6")
        self.formLayout_46.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_6
        )
        self.horizontalLayout_135 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_135.setSpacing(8)
        self.horizontalLayout_135.setObjectName("horizontalLayout_135")
        self.treeCsvBrowseBox_6 = QtWidgets.QLineEdit(self.widget_34)
        self.treeCsvBrowseBox_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_6.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_6.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_6.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_6.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_6.setMaxLength(1000)
        self.treeCsvBrowseBox_6.setObjectName("treeCsvBrowseBox_6")
        self.horizontalLayout_135.addWidget(self.treeCsvBrowseBox_6)
        self.treeCsvBrowseButton_6 = QtWidgets.QPushButton(self.widget_34)
        self.treeCsvBrowseButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_6.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_6.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_6.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_6.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_6.setObjectName("treeCsvBrowseButton_6")
        self.horizontalLayout_135.addWidget(self.treeCsvBrowseButton_6)
        self.formLayout_46.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_135
        )
        self.label_172 = QtWidgets.QLabel(self.widget_34)
        self.label_172.setObjectName("label_172")
        self.formLayout_46.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_172)
        self.horizontalLayout_136 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_136.setSpacing(8)
        self.horizontalLayout_136.setObjectName("horizontalLayout_136")
        self.standDens_6 = QtWidgets.QSpinBox(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_6.sizePolicy().hasHeightForWidth())
        self.standDens_6.setSizePolicy(sizePolicy)
        self.standDens_6.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_6.setMaximum(999)
        self.standDens_6.setSingleStep(50)
        self.standDens_6.setProperty("value", 200)
        self.standDens_6.setObjectName("standDens_6")
        self.horizontalLayout_136.addWidget(self.standDens_6)
        self.label_173 = QtWidgets.QLabel(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_173.sizePolicy().hasHeightForWidth())
        self.label_173.setSizePolicy(sizePolicy)
        self.label_173.setObjectName("label_173")
        self.horizontalLayout_136.addWidget(self.label_173)
        self.formLayout_46.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_136
        )
        self.label_174 = QtWidgets.QLabel(self.widget_34)
        self.label_174.setObjectName("label_174")
        self.formLayout_46.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_174)
        self.horizontalLayout_137 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_137.setSpacing(8)
        self.horizontalLayout_137.setObjectName("horizontalLayout_137")
        self.label_175 = QtWidgets.QLabel(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_175.sizePolicy().hasHeightForWidth())
        self.label_175.setSizePolicy(sizePolicy)
        self.label_175.setObjectName("label_175")
        self.horizontalLayout_137.addWidget(self.label_175)
        self.yearPlanted_6 = QtWidgets.QSpinBox(self.widget_34)
        self.yearPlanted_6.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_6.setObjectName("yearPlanted_6")
        self.horizontalLayout_137.addWidget(self.yearPlanted_6)
        self.formLayout_46.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_137
        )
        self.label_38 = QtWidgets.QLabel(self.widget_34)
        self.label_38.setObjectName("label_38")
        self.formLayout_46.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_38)
        self.speciesName_6 = QtWidgets.QLineEdit(self.widget_34)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_6.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_6.setSizePolicy(sizePolicy)
        self.speciesName_6.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_6.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_6.setObjectName("speciesName_6")
        self.formLayout_46.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_6
        )
        self.gridLayout_39.addLayout(self.formLayout_46, 1, 0, 1, 1)
        self.verticalLayout_88.addWidget(self.widget_34)
        self.stackedWidget.addWidget(self.treeMgmtPage_6)
        self.treeThinPage_6 = QtWidgets.QWidget()
        self.treeThinPage_6.setObjectName("treeThinPage_6")
        self.verticalLayout_90 = QtWidgets.QVBoxLayout(self.treeThinPage_6)
        self.verticalLayout_90.setObjectName("verticalLayout_90")
        self.widget_35 = QtWidgets.QWidget(self.treeThinPage_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_35.sizePolicy().hasHeightForWidth())
        self.widget_35.setSizePolicy(sizePolicy)
        self.widget_35.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_35.setObjectName("widget_35")
        self.gridLayout_40 = QtWidgets.QGridLayout(self.widget_35)
        self.gridLayout_40.setObjectName("gridLayout_40")
        self.label_176 = QtWidgets.QLabel(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_176.sizePolicy().hasHeightForWidth())
        self.label_176.setSizePolicy(sizePolicy)
        self.label_176.setMinimumSize(QtCore.QSize(0, 30))
        self.label_176.setObjectName("label_176")
        self.gridLayout_40.addWidget(self.label_176, 0, 0, 1, 1)
        self.formLayout_47 = QtWidgets.QFormLayout()
        self.formLayout_47.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_47.setVerticalSpacing(15)
        self.formLayout_47.setObjectName("formLayout_47")
        spacerItem39 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_47.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem39)
        self.label_177 = QtWidgets.QLabel(self.widget_35)
        self.label_177.setObjectName("label_177")
        self.formLayout_47.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_177)
        self.verticalLayout_89 = QtWidgets.QVBoxLayout()
        self.verticalLayout_89.setObjectName("verticalLayout_89")
        self.thinNever_6 = QtWidgets.QRadioButton(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_6.sizePolicy().hasHeightForWidth())
        self.thinNever_6.setSizePolicy(sizePolicy)
        self.thinNever_6.setObjectName("thinNever_6")
        self.verticalLayout_89.addWidget(self.thinNever_6)
        self.horizontalLayout_138 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_138.setSpacing(8)
        self.horizontalLayout_138.setObjectName("horizontalLayout_138")
        self.thinInterval_6 = QtWidgets.QRadioButton(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_6.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_6.setSizePolicy(sizePolicy)
        self.thinInterval_6.setObjectName("thinInterval_6")
        self.horizontalLayout_138.addWidget(self.thinInterval_6)
        self.thinFreq_6 = QtWidgets.QSpinBox(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_6.sizePolicy().hasHeightForWidth())
        self.thinFreq_6.setSizePolicy(sizePolicy)
        self.thinFreq_6.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_6.setMinimum(1)
        self.thinFreq_6.setSingleStep(10)
        self.thinFreq_6.setProperty("value", 1)
        self.thinFreq_6.setObjectName("thinFreq_6")
        self.horizontalLayout_138.addWidget(self.thinFreq_6)
        self.thinFreqLabel_6 = QtWidgets.QLabel(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_6.setSizePolicy(sizePolicy)
        self.thinFreqLabel_6.setObjectName("thinFreqLabel_6")
        self.horizontalLayout_138.addWidget(self.thinFreqLabel_6)
        self.verticalLayout_89.addLayout(self.horizontalLayout_138)
        self.thinCustom_6 = QtWidgets.QRadioButton(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_6.sizePolicy().hasHeightForWidth())
        self.thinCustom_6.setSizePolicy(sizePolicy)
        self.thinCustom_6.setObjectName("thinCustom_6")
        self.verticalLayout_89.addWidget(self.thinCustom_6)
        self.formLayout_47.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_89
        )
        self.thinCustomText_6 = QtWidgets.QLabel(self.widget_35)
        self.thinCustomText_6.setObjectName("thinCustomText_6")
        self.formLayout_47.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_6
        )
        self.thinCustomLine_6 = QtWidgets.QLineEdit(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_6.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_6.setSizePolicy(sizePolicy)
        self.thinCustomLine_6.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_6.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_6.setObjectName("thinCustomLine_6")
        self.formLayout_47.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_6
        )
        self.treesThinnedText_6 = QtWidgets.QLabel(self.widget_35)
        self.treesThinnedText_6.setObjectName("treesThinnedText_6")
        self.formLayout_47.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_6
        )
        self.horizontalLayout_139 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_139.setSpacing(8)
        self.horizontalLayout_139.setObjectName("horizontalLayout_139")
        self.treesThinned_6 = QtWidgets.QSpinBox(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_6.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_6.setSizePolicy(sizePolicy)
        self.treesThinned_6.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_6.setMaximum(100)
        self.treesThinned_6.setSingleStep(10)
        self.treesThinned_6.setProperty("value", 50)
        self.treesThinned_6.setObjectName("treesThinned_6")
        self.horizontalLayout_139.addWidget(self.treesThinned_6)
        self.treesThinnedLabel_6 = QtWidgets.QLabel(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_6.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_6.setObjectName("treesThinnedLabel_6")
        self.horizontalLayout_139.addWidget(self.treesThinnedLabel_6)
        self.formLayout_47.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_139
        )
        self.thinnedLeftText_6 = QtWidgets.QLabel(self.widget_35)
        self.thinnedLeftText_6.setObjectName("thinnedLeftText_6")
        self.formLayout_47.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_6
        )
        self.thinnedStemsLeftText_6 = QtWidgets.QLabel(self.widget_35)
        self.thinnedStemsLeftText_6.setObjectName("thinnedStemsLeftText_6")
        self.formLayout_47.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_6
        )
        self.horizontalLayout_140 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_140.setSpacing(8)
        self.horizontalLayout_140.setObjectName("horizontalLayout_140")
        self.thinnedStemsLeft_6 = QtWidgets.QSpinBox(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_6.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_6.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_6.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_6.setMaximum(100)
        self.thinnedStemsLeft_6.setSingleStep(10)
        self.thinnedStemsLeft_6.setObjectName("thinnedStemsLeft_6")
        self.horizontalLayout_140.addWidget(self.thinnedStemsLeft_6)
        self.thinnedStemsLeftLabel_6 = QtWidgets.QLabel(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_6.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_6.setObjectName("thinnedStemsLeftLabel_6")
        self.horizontalLayout_140.addWidget(self.thinnedStemsLeftLabel_6)
        self.formLayout_47.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_140
        )
        self.thinnedBranchesLeftText_6 = QtWidgets.QLabel(self.widget_35)
        self.thinnedBranchesLeftText_6.setObjectName("thinnedBranchesLeftText_6")
        self.formLayout_47.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_6
        )
        self.horizontalLayout_141 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_141.setSpacing(8)
        self.horizontalLayout_141.setObjectName("horizontalLayout_141")
        self.thinnedBranchesLeft_6 = QtWidgets.QSpinBox(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_6.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_6.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_6.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_6.setMaximum(100)
        self.thinnedBranchesLeft_6.setSingleStep(10)
        self.thinnedBranchesLeft_6.setObjectName("thinnedBranchesLeft_6")
        self.horizontalLayout_141.addWidget(self.thinnedBranchesLeft_6)
        self.thinnedBranchesLeftLabel_6 = QtWidgets.QLabel(self.widget_35)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_6.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_6.setObjectName("thinnedBranchesLeftLabel_6")
        self.horizontalLayout_141.addWidget(self.thinnedBranchesLeftLabel_6)
        self.formLayout_47.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_141
        )
        self.gridLayout_40.addLayout(self.formLayout_47, 1, 0, 1, 1)
        self.verticalLayout_90.addWidget(self.widget_35)
        self.stackedWidget.addWidget(self.treeThinPage_6)
        self.treeMortPage_6 = QtWidgets.QWidget()
        self.treeMortPage_6.setObjectName("treeMortPage_6")
        self.verticalLayout_91 = QtWidgets.QVBoxLayout(self.treeMortPage_6)
        self.verticalLayout_91.setObjectName("verticalLayout_91")
        self.widget_36 = QtWidgets.QWidget(self.treeMortPage_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_36.sizePolicy().hasHeightForWidth())
        self.widget_36.setSizePolicy(sizePolicy)
        self.widget_36.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_36.setObjectName("widget_36")
        self.gridLayout_41 = QtWidgets.QGridLayout(self.widget_36)
        self.gridLayout_41.setObjectName("gridLayout_41")
        self.label_178 = QtWidgets.QLabel(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_178.sizePolicy().hasHeightForWidth())
        self.label_178.setSizePolicy(sizePolicy)
        self.label_178.setMinimumSize(QtCore.QSize(0, 30))
        self.label_178.setObjectName("label_178")
        self.gridLayout_41.addWidget(self.label_178, 0, 0, 1, 1)
        self.formLayout_48 = QtWidgets.QFormLayout()
        self.formLayout_48.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_48.setVerticalSpacing(15)
        self.formLayout_48.setObjectName("formLayout_48")
        spacerItem40 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_48.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem40)
        self.label_179 = QtWidgets.QLabel(self.widget_36)
        self.label_179.setObjectName("label_179")
        self.formLayout_48.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_179)
        self.horizontalLayout_142 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_142.setSpacing(8)
        self.horizontalLayout_142.setObjectName("horizontalLayout_142")
        self.treesDead_6 = QtWidgets.QSpinBox(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_6.sizePolicy().hasHeightForWidth())
        self.treesDead_6.setSizePolicy(sizePolicy)
        self.treesDead_6.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_6.setMaximum(100)
        self.treesDead_6.setSingleStep(10)
        self.treesDead_6.setProperty("value", 1)
        self.treesDead_6.setObjectName("treesDead_6")
        self.horizontalLayout_142.addWidget(self.treesDead_6)
        self.label_180 = QtWidgets.QLabel(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_180.sizePolicy().hasHeightForWidth())
        self.label_180.setSizePolicy(sizePolicy)
        self.label_180.setObjectName("label_180")
        self.horizontalLayout_142.addWidget(self.label_180)
        self.formLayout_48.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_142
        )
        self.deadLeftText_6 = QtWidgets.QLabel(self.widget_36)
        self.deadLeftText_6.setObjectName("deadLeftText_6")
        self.formLayout_48.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_6
        )
        self.deadStemsLeftText_6 = QtWidgets.QLabel(self.widget_36)
        self.deadStemsLeftText_6.setObjectName("deadStemsLeftText_6")
        self.formLayout_48.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_6
        )
        self.horizontalLayout_143 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_143.setSpacing(8)
        self.horizontalLayout_143.setObjectName("horizontalLayout_143")
        self.deadStemsLeft_6 = QtWidgets.QSpinBox(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_6.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_6.setSizePolicy(sizePolicy)
        self.deadStemsLeft_6.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_6.setMaximum(100)
        self.deadStemsLeft_6.setSingleStep(10)
        self.deadStemsLeft_6.setObjectName("deadStemsLeft_6")
        self.horizontalLayout_143.addWidget(self.deadStemsLeft_6)
        self.deadStemsLeftLabel_6 = QtWidgets.QLabel(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_6.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_6.setObjectName("deadStemsLeftLabel_6")
        self.horizontalLayout_143.addWidget(self.deadStemsLeftLabel_6)
        self.formLayout_48.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_143
        )
        self.deadBranchesLeftText_6 = QtWidgets.QLabel(self.widget_36)
        self.deadBranchesLeftText_6.setObjectName("deadBranchesLeftText_6")
        self.formLayout_48.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_6
        )
        self.horizontalLayout_144 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_144.setSpacing(8)
        self.horizontalLayout_144.setObjectName("horizontalLayout_144")
        self.deadBranchesLeft_6 = QtWidgets.QSpinBox(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_6.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_6.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_6.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_6.setMaximum(100)
        self.deadBranchesLeft_6.setSingleStep(10)
        self.deadBranchesLeft_6.setObjectName("deadBranchesLeft_6")
        self.horizontalLayout_144.addWidget(self.deadBranchesLeft_6)
        self.deadBranchesLeftLabel_6 = QtWidgets.QLabel(self.widget_36)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_6.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_6.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_6.setObjectName("deadBranchesLeftLabel_6")
        self.horizontalLayout_144.addWidget(self.deadBranchesLeftLabel_6)
        self.formLayout_48.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_144
        )
        self.gridLayout_41.addLayout(self.formLayout_48, 1, 0, 1, 1)
        self.verticalLayout_91.addWidget(self.widget_36)
        self.stackedWidget.addWidget(self.treeMortPage_6)
        self.treeGrowthPage_6 = QtWidgets.QWidget()
        self.treeGrowthPage_6.setObjectName("treeGrowthPage_6")
        self.verticalLayout_93 = QtWidgets.QVBoxLayout(self.treeGrowthPage_6)
        self.verticalLayout_93.setObjectName("verticalLayout_93")
        self.widget_37 = QtWidgets.QWidget(self.treeGrowthPage_6)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_37.sizePolicy().hasHeightForWidth())
        self.widget_37.setSizePolicy(sizePolicy)
        self.widget_37.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_37.setObjectName("widget_37")
        self.gridLayout_42 = QtWidgets.QGridLayout(self.widget_37)
        self.gridLayout_42.setObjectName("gridLayout_42")
        self.formLayout_49 = QtWidgets.QFormLayout()
        self.formLayout_49.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_49.setVerticalSpacing(15)
        self.formLayout_49.setObjectName("formLayout_49")
        spacerItem41 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_49.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem41)
        self.verticalLayout_92 = QtWidgets.QVBoxLayout()
        self.verticalLayout_92.setSpacing(8)
        self.verticalLayout_92.setObjectName("verticalLayout_92")
        self.growthCsvButton_6 = QtWidgets.QRadioButton(self.widget_37)
        self.growthCsvButton_6.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_6.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_6.setSizePolicy(sizePolicy)
        self.growthCsvButton_6.setObjectName("growthCsvButton_6")
        self.verticalLayout_92.addWidget(self.growthCsvButton_6)
        self.growthCustomButton_6 = QtWidgets.QRadioButton(self.widget_37)
        self.growthCustomButton_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_6.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_6.setSizePolicy(sizePolicy)
        self.growthCustomButton_6.setObjectName("growthCustomButton_6")
        self.verticalLayout_92.addWidget(self.growthCustomButton_6)
        self.formLayout_49.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_92
        )
        self.growthCsvLabel_6 = QtWidgets.QLabel(self.widget_37)
        self.growthCsvLabel_6.setEnabled(True)
        self.growthCsvLabel_6.setObjectName("growthCsvLabel_6")
        self.formLayout_49.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_6
        )
        self.horizontalLayout_145 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_145.setSpacing(8)
        self.horizontalLayout_145.setObjectName("horizontalLayout_145")
        self.growthCsvBrowseBox_6 = QtWidgets.QLineEdit(self.widget_37)
        self.growthCsvBrowseBox_6.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_6.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_6.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_6.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_6.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_6.setMaxLength(1000)
        self.growthCsvBrowseBox_6.setObjectName("growthCsvBrowseBox_6")
        self.horizontalLayout_145.addWidget(self.growthCsvBrowseBox_6)
        self.growthCsvBrowseButton_6 = QtWidgets.QPushButton(self.widget_37)
        self.growthCsvBrowseButton_6.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_6.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_6.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_6.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_6.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_6.setObjectName("growthCsvBrowseButton_6")
        self.horizontalLayout_145.addWidget(self.growthCsvBrowseButton_6)
        self.formLayout_49.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_145
        )
        self.label_181 = QtWidgets.QLabel(self.widget_37)
        self.label_181.setObjectName("label_181")
        self.formLayout_49.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_181)
        self.allom_6 = QtWidgets.QComboBox(self.widget_37)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_6.sizePolicy().hasHeightForWidth())
        self.allom_6.setSizePolicy(sizePolicy)
        self.allom_6.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_6.setObjectName("allom_6")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.setItemText(1, "")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.setItemText(7, "")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.allom_6.setItemText(12, "")
        self.allom_6.addItem("")
        self.allom_6.addItem("")
        self.formLayout_49.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_6)
        self.gridLayout_42.addLayout(self.formLayout_49, 1, 0, 1, 1)
        self.label_182 = QtWidgets.QLabel(self.widget_37)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_182.sizePolicy().hasHeightForWidth())
        self.label_182.setSizePolicy(sizePolicy)
        self.label_182.setMinimumSize(QtCore.QSize(0, 30))
        self.label_182.setObjectName("label_182")
        self.gridLayout_42.addWidget(self.label_182, 0, 0, 1, 1)
        self.verticalLayout_93.addWidget(self.widget_37)
        self.stackedWidget.addWidget(self.treeGrowthPage_6)
        self.treeMgmtPage_7 = QtWidgets.QWidget()
        self.treeMgmtPage_7.setObjectName("treeMgmtPage_7")
        self.verticalLayout_95 = QtWidgets.QVBoxLayout(self.treeMgmtPage_7)
        self.verticalLayout_95.setObjectName("verticalLayout_95")
        self.widget_38 = QtWidgets.QWidget(self.treeMgmtPage_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_38.sizePolicy().hasHeightForWidth())
        self.widget_38.setSizePolicy(sizePolicy)
        self.widget_38.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_38.setObjectName("widget_38")
        self.gridLayout_43 = QtWidgets.QGridLayout(self.widget_38)
        self.gridLayout_43.setObjectName("gridLayout_43")
        self.label_183 = QtWidgets.QLabel(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_183.sizePolicy().hasHeightForWidth())
        self.label_183.setSizePolicy(sizePolicy)
        self.label_183.setMinimumSize(QtCore.QSize(0, 30))
        self.label_183.setObjectName("label_183")
        self.gridLayout_43.addWidget(self.label_183, 0, 0, 1, 1)
        self.formLayout_50 = QtWidgets.QFormLayout()
        self.formLayout_50.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_50.setVerticalSpacing(15)
        self.formLayout_50.setObjectName("formLayout_50")
        spacerItem42 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_50.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem42)
        self.label_184 = QtWidgets.QLabel(self.widget_38)
        self.label_184.setObjectName("label_184")
        self.formLayout_50.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_184)
        self.treeType_7 = QtWidgets.QComboBox(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_7.sizePolicy().hasHeightForWidth())
        self.treeType_7.setSizePolicy(sizePolicy)
        self.treeType_7.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_7.setObjectName("treeType_7")
        self.treeType_7.addItem("")
        self.treeType_7.addItem("")
        self.treeType_7.setItemText(1, "")
        self.treeType_7.addItem("")
        self.treeType_7.addItem("")
        self.treeType_7.addItem("")
        self.treeType_7.setItemText(4, "")
        self.treeType_7.addItem("")
        self.treeType_7.addItem("")
        self.formLayout_50.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_7
        )
        self.treeInputType_7 = QtWidgets.QLabel(self.widget_38)
        self.treeInputType_7.setEnabled(False)
        self.treeInputType_7.setObjectName("treeInputType_7")
        self.formLayout_50.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_7
        )
        self.verticalLayout_94 = QtWidgets.QVBoxLayout()
        self.verticalLayout_94.setSpacing(8)
        self.verticalLayout_94.setObjectName("verticalLayout_94")
        self.treeCsvButton_7 = QtWidgets.QRadioButton(self.widget_38)
        self.treeCsvButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_7.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_7.setSizePolicy(sizePolicy)
        self.treeCsvButton_7.setObjectName("treeCsvButton_7")
        self.verticalLayout_94.addWidget(self.treeCsvButton_7)
        self.treeCustomButton_7 = QtWidgets.QRadioButton(self.widget_38)
        self.treeCustomButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_7.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_7.setSizePolicy(sizePolicy)
        self.treeCustomButton_7.setObjectName("treeCustomButton_7")
        self.verticalLayout_94.addWidget(self.treeCustomButton_7)
        self.formLayout_50.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_94
        )
        self.treeCsvLabel_7 = QtWidgets.QLabel(self.widget_38)
        self.treeCsvLabel_7.setEnabled(False)
        self.treeCsvLabel_7.setObjectName("treeCsvLabel_7")
        self.formLayout_50.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_7
        )
        self.horizontalLayout_146 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_146.setSpacing(8)
        self.horizontalLayout_146.setObjectName("horizontalLayout_146")
        self.treeCsvBrowseBox_7 = QtWidgets.QLineEdit(self.widget_38)
        self.treeCsvBrowseBox_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_7.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_7.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_7.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_7.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_7.setMaxLength(1000)
        self.treeCsvBrowseBox_7.setObjectName("treeCsvBrowseBox_7")
        self.horizontalLayout_146.addWidget(self.treeCsvBrowseBox_7)
        self.treeCsvBrowseButton_7 = QtWidgets.QPushButton(self.widget_38)
        self.treeCsvBrowseButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_7.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_7.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_7.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_7.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_7.setObjectName("treeCsvBrowseButton_7")
        self.horizontalLayout_146.addWidget(self.treeCsvBrowseButton_7)
        self.formLayout_50.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_146
        )
        self.label_185 = QtWidgets.QLabel(self.widget_38)
        self.label_185.setObjectName("label_185")
        self.formLayout_50.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_185)
        self.horizontalLayout_147 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_147.setSpacing(8)
        self.horizontalLayout_147.setObjectName("horizontalLayout_147")
        self.standDens_7 = QtWidgets.QSpinBox(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_7.sizePolicy().hasHeightForWidth())
        self.standDens_7.setSizePolicy(sizePolicy)
        self.standDens_7.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_7.setMaximum(999)
        self.standDens_7.setSingleStep(50)
        self.standDens_7.setProperty("value", 200)
        self.standDens_7.setObjectName("standDens_7")
        self.horizontalLayout_147.addWidget(self.standDens_7)
        self.label_186 = QtWidgets.QLabel(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_186.sizePolicy().hasHeightForWidth())
        self.label_186.setSizePolicy(sizePolicy)
        self.label_186.setObjectName("label_186")
        self.horizontalLayout_147.addWidget(self.label_186)
        self.formLayout_50.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_147
        )
        self.label_187 = QtWidgets.QLabel(self.widget_38)
        self.label_187.setObjectName("label_187")
        self.formLayout_50.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_187)
        self.horizontalLayout_148 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_148.setSpacing(8)
        self.horizontalLayout_148.setObjectName("horizontalLayout_148")
        self.label_188 = QtWidgets.QLabel(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_188.sizePolicy().hasHeightForWidth())
        self.label_188.setSizePolicy(sizePolicy)
        self.label_188.setObjectName("label_188")
        self.horizontalLayout_148.addWidget(self.label_188)
        self.yearPlanted_7 = QtWidgets.QSpinBox(self.widget_38)
        self.yearPlanted_7.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_7.setObjectName("yearPlanted_7")
        self.horizontalLayout_148.addWidget(self.yearPlanted_7)
        self.formLayout_50.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_148
        )
        self.label_27 = QtWidgets.QLabel(self.widget_38)
        self.label_27.setObjectName("label_27")
        self.formLayout_50.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.speciesName_7 = QtWidgets.QLineEdit(self.widget_38)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_7.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_7.setSizePolicy(sizePolicy)
        self.speciesName_7.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_7.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_7.setObjectName("speciesName_7")
        self.formLayout_50.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_7
        )
        self.gridLayout_43.addLayout(self.formLayout_50, 1, 0, 1, 1)
        self.verticalLayout_95.addWidget(self.widget_38)
        self.stackedWidget.addWidget(self.treeMgmtPage_7)
        self.treeThinPage_7 = QtWidgets.QWidget()
        self.treeThinPage_7.setObjectName("treeThinPage_7")
        self.verticalLayout_97 = QtWidgets.QVBoxLayout(self.treeThinPage_7)
        self.verticalLayout_97.setObjectName("verticalLayout_97")
        self.widget_39 = QtWidgets.QWidget(self.treeThinPage_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_39.sizePolicy().hasHeightForWidth())
        self.widget_39.setSizePolicy(sizePolicy)
        self.widget_39.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_39.setObjectName("widget_39")
        self.gridLayout_44 = QtWidgets.QGridLayout(self.widget_39)
        self.gridLayout_44.setObjectName("gridLayout_44")
        self.label_189 = QtWidgets.QLabel(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_189.sizePolicy().hasHeightForWidth())
        self.label_189.setSizePolicy(sizePolicy)
        self.label_189.setMinimumSize(QtCore.QSize(0, 30))
        self.label_189.setObjectName("label_189")
        self.gridLayout_44.addWidget(self.label_189, 0, 0, 1, 1)
        self.formLayout_51 = QtWidgets.QFormLayout()
        self.formLayout_51.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_51.setVerticalSpacing(15)
        self.formLayout_51.setObjectName("formLayout_51")
        spacerItem43 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_51.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem43)
        self.label_190 = QtWidgets.QLabel(self.widget_39)
        self.label_190.setObjectName("label_190")
        self.formLayout_51.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_190)
        self.verticalLayout_96 = QtWidgets.QVBoxLayout()
        self.verticalLayout_96.setObjectName("verticalLayout_96")
        self.thinNever_7 = QtWidgets.QRadioButton(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_7.sizePolicy().hasHeightForWidth())
        self.thinNever_7.setSizePolicy(sizePolicy)
        self.thinNever_7.setObjectName("thinNever_7")
        self.verticalLayout_96.addWidget(self.thinNever_7)
        self.horizontalLayout_149 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_149.setSpacing(8)
        self.horizontalLayout_149.setObjectName("horizontalLayout_149")
        self.thinInterval_7 = QtWidgets.QRadioButton(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_7.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_7.setSizePolicy(sizePolicy)
        self.thinInterval_7.setObjectName("thinInterval_7")
        self.horizontalLayout_149.addWidget(self.thinInterval_7)
        self.thinFreq_7 = QtWidgets.QSpinBox(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_7.sizePolicy().hasHeightForWidth())
        self.thinFreq_7.setSizePolicy(sizePolicy)
        self.thinFreq_7.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_7.setMinimum(1)
        self.thinFreq_7.setSingleStep(10)
        self.thinFreq_7.setProperty("value", 1)
        self.thinFreq_7.setObjectName("thinFreq_7")
        self.horizontalLayout_149.addWidget(self.thinFreq_7)
        self.thinFreqLabel_7 = QtWidgets.QLabel(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_7.setSizePolicy(sizePolicy)
        self.thinFreqLabel_7.setObjectName("thinFreqLabel_7")
        self.horizontalLayout_149.addWidget(self.thinFreqLabel_7)
        self.verticalLayout_96.addLayout(self.horizontalLayout_149)
        self.thinCustom_7 = QtWidgets.QRadioButton(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_7.sizePolicy().hasHeightForWidth())
        self.thinCustom_7.setSizePolicy(sizePolicy)
        self.thinCustom_7.setObjectName("thinCustom_7")
        self.verticalLayout_96.addWidget(self.thinCustom_7)
        self.formLayout_51.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_96
        )
        self.thinCustomText_7 = QtWidgets.QLabel(self.widget_39)
        self.thinCustomText_7.setObjectName("thinCustomText_7")
        self.formLayout_51.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_7
        )
        self.thinCustomLine_7 = QtWidgets.QLineEdit(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_7.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_7.setSizePolicy(sizePolicy)
        self.thinCustomLine_7.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_7.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_7.setObjectName("thinCustomLine_7")
        self.formLayout_51.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_7
        )
        self.treesThinnedText_7 = QtWidgets.QLabel(self.widget_39)
        self.treesThinnedText_7.setObjectName("treesThinnedText_7")
        self.formLayout_51.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_7
        )
        self.horizontalLayout_150 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_150.setSpacing(8)
        self.horizontalLayout_150.setObjectName("horizontalLayout_150")
        self.treesThinned_7 = QtWidgets.QSpinBox(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_7.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_7.setSizePolicy(sizePolicy)
        self.treesThinned_7.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_7.setMaximum(100)
        self.treesThinned_7.setSingleStep(10)
        self.treesThinned_7.setProperty("value", 50)
        self.treesThinned_7.setObjectName("treesThinned_7")
        self.horizontalLayout_150.addWidget(self.treesThinned_7)
        self.treesThinnedLabel_7 = QtWidgets.QLabel(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_7.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_7.setObjectName("treesThinnedLabel_7")
        self.horizontalLayout_150.addWidget(self.treesThinnedLabel_7)
        self.formLayout_51.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_150
        )
        self.thinnedLeftText_7 = QtWidgets.QLabel(self.widget_39)
        self.thinnedLeftText_7.setObjectName("thinnedLeftText_7")
        self.formLayout_51.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_7
        )
        self.thinnedStemsLeftText_7 = QtWidgets.QLabel(self.widget_39)
        self.thinnedStemsLeftText_7.setObjectName("thinnedStemsLeftText_7")
        self.formLayout_51.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_7
        )
        self.horizontalLayout_151 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_151.setSpacing(8)
        self.horizontalLayout_151.setObjectName("horizontalLayout_151")
        self.thinnedStemsLeft_7 = QtWidgets.QSpinBox(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_7.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_7.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_7.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_7.setMaximum(100)
        self.thinnedStemsLeft_7.setSingleStep(10)
        self.thinnedStemsLeft_7.setObjectName("thinnedStemsLeft_7")
        self.horizontalLayout_151.addWidget(self.thinnedStemsLeft_7)
        self.thinnedStemsLeftLabel_7 = QtWidgets.QLabel(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_7.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_7.setObjectName("thinnedStemsLeftLabel_7")
        self.horizontalLayout_151.addWidget(self.thinnedStemsLeftLabel_7)
        self.formLayout_51.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_151
        )
        self.thinnedBranchesLeftText_7 = QtWidgets.QLabel(self.widget_39)
        self.thinnedBranchesLeftText_7.setObjectName("thinnedBranchesLeftText_7")
        self.formLayout_51.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_7
        )
        self.horizontalLayout_152 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_152.setSpacing(8)
        self.horizontalLayout_152.setObjectName("horizontalLayout_152")
        self.thinnedBranchesLeft_7 = QtWidgets.QSpinBox(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_7.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_7.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_7.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_7.setMaximum(100)
        self.thinnedBranchesLeft_7.setSingleStep(10)
        self.thinnedBranchesLeft_7.setObjectName("thinnedBranchesLeft_7")
        self.horizontalLayout_152.addWidget(self.thinnedBranchesLeft_7)
        self.thinnedBranchesLeftLabel_7 = QtWidgets.QLabel(self.widget_39)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_7.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_7.setObjectName("thinnedBranchesLeftLabel_7")
        self.horizontalLayout_152.addWidget(self.thinnedBranchesLeftLabel_7)
        self.formLayout_51.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_152
        )
        self.gridLayout_44.addLayout(self.formLayout_51, 1, 0, 1, 1)
        self.verticalLayout_97.addWidget(self.widget_39)
        self.stackedWidget.addWidget(self.treeThinPage_7)
        self.treeMortPage_7 = QtWidgets.QWidget()
        self.treeMortPage_7.setObjectName("treeMortPage_7")
        self.verticalLayout_98 = QtWidgets.QVBoxLayout(self.treeMortPage_7)
        self.verticalLayout_98.setObjectName("verticalLayout_98")
        self.widget_40 = QtWidgets.QWidget(self.treeMortPage_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_40.sizePolicy().hasHeightForWidth())
        self.widget_40.setSizePolicy(sizePolicy)
        self.widget_40.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_40.setObjectName("widget_40")
        self.gridLayout_45 = QtWidgets.QGridLayout(self.widget_40)
        self.gridLayout_45.setObjectName("gridLayout_45")
        self.label_191 = QtWidgets.QLabel(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_191.sizePolicy().hasHeightForWidth())
        self.label_191.setSizePolicy(sizePolicy)
        self.label_191.setMinimumSize(QtCore.QSize(0, 30))
        self.label_191.setObjectName("label_191")
        self.gridLayout_45.addWidget(self.label_191, 0, 0, 1, 1)
        self.formLayout_52 = QtWidgets.QFormLayout()
        self.formLayout_52.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_52.setVerticalSpacing(15)
        self.formLayout_52.setObjectName("formLayout_52")
        spacerItem44 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_52.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem44)
        self.label_192 = QtWidgets.QLabel(self.widget_40)
        self.label_192.setObjectName("label_192")
        self.formLayout_52.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_192)
        self.horizontalLayout_153 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_153.setSpacing(8)
        self.horizontalLayout_153.setObjectName("horizontalLayout_153")
        self.treesDead_7 = QtWidgets.QSpinBox(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_7.sizePolicy().hasHeightForWidth())
        self.treesDead_7.setSizePolicy(sizePolicy)
        self.treesDead_7.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_7.setMaximum(100)
        self.treesDead_7.setSingleStep(10)
        self.treesDead_7.setProperty("value", 1)
        self.treesDead_7.setObjectName("treesDead_7")
        self.horizontalLayout_153.addWidget(self.treesDead_7)
        self.label_193 = QtWidgets.QLabel(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_193.sizePolicy().hasHeightForWidth())
        self.label_193.setSizePolicy(sizePolicy)
        self.label_193.setObjectName("label_193")
        self.horizontalLayout_153.addWidget(self.label_193)
        self.formLayout_52.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_153
        )
        self.deadLeftText_7 = QtWidgets.QLabel(self.widget_40)
        self.deadLeftText_7.setObjectName("deadLeftText_7")
        self.formLayout_52.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_7
        )
        self.deadStemsLeftText_7 = QtWidgets.QLabel(self.widget_40)
        self.deadStemsLeftText_7.setObjectName("deadStemsLeftText_7")
        self.formLayout_52.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_7
        )
        self.horizontalLayout_154 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_154.setSpacing(8)
        self.horizontalLayout_154.setObjectName("horizontalLayout_154")
        self.deadStemsLeft_7 = QtWidgets.QSpinBox(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_7.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_7.setSizePolicy(sizePolicy)
        self.deadStemsLeft_7.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_7.setMaximum(100)
        self.deadStemsLeft_7.setSingleStep(10)
        self.deadStemsLeft_7.setObjectName("deadStemsLeft_7")
        self.horizontalLayout_154.addWidget(self.deadStemsLeft_7)
        self.deadStemsLeftLabel_7 = QtWidgets.QLabel(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_7.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_7.setObjectName("deadStemsLeftLabel_7")
        self.horizontalLayout_154.addWidget(self.deadStemsLeftLabel_7)
        self.formLayout_52.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_154
        )
        self.deadBranchesLeftText_7 = QtWidgets.QLabel(self.widget_40)
        self.deadBranchesLeftText_7.setObjectName("deadBranchesLeftText_7")
        self.formLayout_52.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_7
        )
        self.horizontalLayout_155 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_155.setSpacing(8)
        self.horizontalLayout_155.setObjectName("horizontalLayout_155")
        self.deadBranchesLeft_7 = QtWidgets.QSpinBox(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_7.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_7.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_7.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_7.setMaximum(100)
        self.deadBranchesLeft_7.setSingleStep(10)
        self.deadBranchesLeft_7.setObjectName("deadBranchesLeft_7")
        self.horizontalLayout_155.addWidget(self.deadBranchesLeft_7)
        self.deadBranchesLeftLabel_7 = QtWidgets.QLabel(self.widget_40)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_7.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_7.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_7.setObjectName("deadBranchesLeftLabel_7")
        self.horizontalLayout_155.addWidget(self.deadBranchesLeftLabel_7)
        self.formLayout_52.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_155
        )
        self.gridLayout_45.addLayout(self.formLayout_52, 1, 0, 1, 1)
        self.verticalLayout_98.addWidget(self.widget_40)
        self.stackedWidget.addWidget(self.treeMortPage_7)
        self.treeGrowthPage_7 = QtWidgets.QWidget()
        self.treeGrowthPage_7.setObjectName("treeGrowthPage_7")
        self.verticalLayout_100 = QtWidgets.QVBoxLayout(self.treeGrowthPage_7)
        self.verticalLayout_100.setObjectName("verticalLayout_100")
        self.widget_41 = QtWidgets.QWidget(self.treeGrowthPage_7)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_41.sizePolicy().hasHeightForWidth())
        self.widget_41.setSizePolicy(sizePolicy)
        self.widget_41.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_41.setObjectName("widget_41")
        self.gridLayout_46 = QtWidgets.QGridLayout(self.widget_41)
        self.gridLayout_46.setObjectName("gridLayout_46")
        self.formLayout_53 = QtWidgets.QFormLayout()
        self.formLayout_53.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_53.setVerticalSpacing(15)
        self.formLayout_53.setObjectName("formLayout_53")
        spacerItem45 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_53.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem45)
        self.verticalLayout_99 = QtWidgets.QVBoxLayout()
        self.verticalLayout_99.setSpacing(8)
        self.verticalLayout_99.setObjectName("verticalLayout_99")
        self.growthCsvButton_7 = QtWidgets.QRadioButton(self.widget_41)
        self.growthCsvButton_7.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_7.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_7.setSizePolicy(sizePolicy)
        self.growthCsvButton_7.setObjectName("growthCsvButton_7")
        self.verticalLayout_99.addWidget(self.growthCsvButton_7)
        self.growthCustomButton_7 = QtWidgets.QRadioButton(self.widget_41)
        self.growthCustomButton_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_7.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_7.setSizePolicy(sizePolicy)
        self.growthCustomButton_7.setObjectName("growthCustomButton_7")
        self.verticalLayout_99.addWidget(self.growthCustomButton_7)
        self.formLayout_53.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_99
        )
        self.growthCsvLabel_7 = QtWidgets.QLabel(self.widget_41)
        self.growthCsvLabel_7.setEnabled(True)
        self.growthCsvLabel_7.setObjectName("growthCsvLabel_7")
        self.formLayout_53.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_7
        )
        self.horizontalLayout_156 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_156.setSpacing(8)
        self.horizontalLayout_156.setObjectName("horizontalLayout_156")
        self.growthCsvBrowseBox_7 = QtWidgets.QLineEdit(self.widget_41)
        self.growthCsvBrowseBox_7.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_7.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_7.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_7.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_7.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_7.setMaxLength(1000)
        self.growthCsvBrowseBox_7.setObjectName("growthCsvBrowseBox_7")
        self.horizontalLayout_156.addWidget(self.growthCsvBrowseBox_7)
        self.growthCsvBrowseButton_7 = QtWidgets.QPushButton(self.widget_41)
        self.growthCsvBrowseButton_7.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_7.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_7.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_7.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_7.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_7.setObjectName("growthCsvBrowseButton_7")
        self.horizontalLayout_156.addWidget(self.growthCsvBrowseButton_7)
        self.formLayout_53.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_156
        )
        self.label_194 = QtWidgets.QLabel(self.widget_41)
        self.label_194.setObjectName("label_194")
        self.formLayout_53.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_194)
        self.allom_7 = QtWidgets.QComboBox(self.widget_41)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_7.sizePolicy().hasHeightForWidth())
        self.allom_7.setSizePolicy(sizePolicy)
        self.allom_7.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_7.setObjectName("allom_7")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.setItemText(1, "")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.setItemText(7, "")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.allom_7.setItemText(12, "")
        self.allom_7.addItem("")
        self.allom_7.addItem("")
        self.formLayout_53.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_7)
        self.gridLayout_46.addLayout(self.formLayout_53, 1, 0, 1, 1)
        self.label_195 = QtWidgets.QLabel(self.widget_41)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_195.sizePolicy().hasHeightForWidth())
        self.label_195.setSizePolicy(sizePolicy)
        self.label_195.setMinimumSize(QtCore.QSize(0, 30))
        self.label_195.setObjectName("label_195")
        self.gridLayout_46.addWidget(self.label_195, 0, 0, 1, 1)
        self.verticalLayout_100.addWidget(self.widget_41)
        self.stackedWidget.addWidget(self.treeGrowthPage_7)
        self.treeMgmtPage_8 = QtWidgets.QWidget()
        self.treeMgmtPage_8.setObjectName("treeMgmtPage_8")
        self.verticalLayout_102 = QtWidgets.QVBoxLayout(self.treeMgmtPage_8)
        self.verticalLayout_102.setObjectName("verticalLayout_102")
        self.widget_42 = QtWidgets.QWidget(self.treeMgmtPage_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_42.sizePolicy().hasHeightForWidth())
        self.widget_42.setSizePolicy(sizePolicy)
        self.widget_42.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_42.setObjectName("widget_42")
        self.gridLayout_47 = QtWidgets.QGridLayout(self.widget_42)
        self.gridLayout_47.setObjectName("gridLayout_47")
        self.label_196 = QtWidgets.QLabel(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_196.sizePolicy().hasHeightForWidth())
        self.label_196.setSizePolicy(sizePolicy)
        self.label_196.setMinimumSize(QtCore.QSize(0, 30))
        self.label_196.setObjectName("label_196")
        self.gridLayout_47.addWidget(self.label_196, 0, 0, 1, 1)
        self.formLayout_54 = QtWidgets.QFormLayout()
        self.formLayout_54.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_54.setVerticalSpacing(15)
        self.formLayout_54.setObjectName("formLayout_54")
        spacerItem46 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_54.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem46)
        self.label_197 = QtWidgets.QLabel(self.widget_42)
        self.label_197.setObjectName("label_197")
        self.formLayout_54.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_197)
        self.treeType_8 = QtWidgets.QComboBox(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_8.sizePolicy().hasHeightForWidth())
        self.treeType_8.setSizePolicy(sizePolicy)
        self.treeType_8.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_8.setObjectName("treeType_8")
        self.treeType_8.addItem("")
        self.treeType_8.addItem("")
        self.treeType_8.setItemText(1, "")
        self.treeType_8.addItem("")
        self.treeType_8.addItem("")
        self.treeType_8.addItem("")
        self.treeType_8.setItemText(4, "")
        self.treeType_8.addItem("")
        self.treeType_8.addItem("")
        self.formLayout_54.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_8
        )
        self.treeInputType_8 = QtWidgets.QLabel(self.widget_42)
        self.treeInputType_8.setEnabled(False)
        self.treeInputType_8.setObjectName("treeInputType_8")
        self.formLayout_54.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_8
        )
        self.verticalLayout_101 = QtWidgets.QVBoxLayout()
        self.verticalLayout_101.setSpacing(8)
        self.verticalLayout_101.setObjectName("verticalLayout_101")
        self.treeCsvButton_8 = QtWidgets.QRadioButton(self.widget_42)
        self.treeCsvButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_8.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_8.setSizePolicy(sizePolicy)
        self.treeCsvButton_8.setObjectName("treeCsvButton_8")
        self.verticalLayout_101.addWidget(self.treeCsvButton_8)
        self.treeCustomButton_8 = QtWidgets.QRadioButton(self.widget_42)
        self.treeCustomButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_8.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_8.setSizePolicy(sizePolicy)
        self.treeCustomButton_8.setObjectName("treeCustomButton_8")
        self.verticalLayout_101.addWidget(self.treeCustomButton_8)
        self.formLayout_54.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_101
        )
        self.treeCsvLabel_8 = QtWidgets.QLabel(self.widget_42)
        self.treeCsvLabel_8.setEnabled(False)
        self.treeCsvLabel_8.setObjectName("treeCsvLabel_8")
        self.formLayout_54.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_8
        )
        self.horizontalLayout_157 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_157.setSpacing(8)
        self.horizontalLayout_157.setObjectName("horizontalLayout_157")
        self.treeCsvBrowseBox_8 = QtWidgets.QLineEdit(self.widget_42)
        self.treeCsvBrowseBox_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_8.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_8.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_8.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_8.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_8.setMaxLength(1000)
        self.treeCsvBrowseBox_8.setObjectName("treeCsvBrowseBox_8")
        self.horizontalLayout_157.addWidget(self.treeCsvBrowseBox_8)
        self.treeCsvBrowseButton_8 = QtWidgets.QPushButton(self.widget_42)
        self.treeCsvBrowseButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_8.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_8.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_8.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_8.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_8.setObjectName("treeCsvBrowseButton_8")
        self.horizontalLayout_157.addWidget(self.treeCsvBrowseButton_8)
        self.formLayout_54.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_157
        )
        self.horizontalLayout_158 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_158.setSpacing(8)
        self.horizontalLayout_158.setObjectName("horizontalLayout_158")
        self.standDens_8 = QtWidgets.QSpinBox(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_8.sizePolicy().hasHeightForWidth())
        self.standDens_8.setSizePolicy(sizePolicy)
        self.standDens_8.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_8.setMaximum(999)
        self.standDens_8.setSingleStep(50)
        self.standDens_8.setProperty("value", 200)
        self.standDens_8.setObjectName("standDens_8")
        self.horizontalLayout_158.addWidget(self.standDens_8)
        self.label_199 = QtWidgets.QLabel(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_199.sizePolicy().hasHeightForWidth())
        self.label_199.setSizePolicy(sizePolicy)
        self.label_199.setObjectName("label_199")
        self.horizontalLayout_158.addWidget(self.label_199)
        self.formLayout_54.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_158
        )
        self.label_200 = QtWidgets.QLabel(self.widget_42)
        self.label_200.setObjectName("label_200")
        self.formLayout_54.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_200)
        self.horizontalLayout_159 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_159.setSpacing(8)
        self.horizontalLayout_159.setObjectName("horizontalLayout_159")
        self.label_201 = QtWidgets.QLabel(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_201.sizePolicy().hasHeightForWidth())
        self.label_201.setSizePolicy(sizePolicy)
        self.label_201.setObjectName("label_201")
        self.horizontalLayout_159.addWidget(self.label_201)
        self.yearPlanted_8 = QtWidgets.QSpinBox(self.widget_42)
        self.yearPlanted_8.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_8.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_8.setObjectName("yearPlanted_8")
        self.horizontalLayout_159.addWidget(self.yearPlanted_8)
        self.formLayout_54.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_159
        )
        self.label_198 = QtWidgets.QLabel(self.widget_42)
        self.label_198.setObjectName("label_198")
        self.formLayout_54.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_198)
        self.label_16 = QtWidgets.QLabel(self.widget_42)
        self.label_16.setObjectName("label_16")
        self.formLayout_54.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.speciesName_8 = QtWidgets.QLineEdit(self.widget_42)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_8.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_8.setSizePolicy(sizePolicy)
        self.speciesName_8.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_8.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_8.setObjectName("speciesName_8")
        self.formLayout_54.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_8
        )
        self.gridLayout_47.addLayout(self.formLayout_54, 1, 0, 1, 1)
        self.verticalLayout_102.addWidget(self.widget_42)
        self.stackedWidget.addWidget(self.treeMgmtPage_8)
        self.treeThinPage_8 = QtWidgets.QWidget()
        self.treeThinPage_8.setObjectName("treeThinPage_8")
        self.verticalLayout_104 = QtWidgets.QVBoxLayout(self.treeThinPage_8)
        self.verticalLayout_104.setObjectName("verticalLayout_104")
        self.widget_43 = QtWidgets.QWidget(self.treeThinPage_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_43.sizePolicy().hasHeightForWidth())
        self.widget_43.setSizePolicy(sizePolicy)
        self.widget_43.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_43.setObjectName("widget_43")
        self.gridLayout_48 = QtWidgets.QGridLayout(self.widget_43)
        self.gridLayout_48.setObjectName("gridLayout_48")
        self.label_202 = QtWidgets.QLabel(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_202.sizePolicy().hasHeightForWidth())
        self.label_202.setSizePolicy(sizePolicy)
        self.label_202.setMinimumSize(QtCore.QSize(0, 30))
        self.label_202.setObjectName("label_202")
        self.gridLayout_48.addWidget(self.label_202, 0, 0, 1, 1)
        self.formLayout_55 = QtWidgets.QFormLayout()
        self.formLayout_55.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_55.setVerticalSpacing(15)
        self.formLayout_55.setObjectName("formLayout_55")
        spacerItem47 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_55.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem47)
        self.label_203 = QtWidgets.QLabel(self.widget_43)
        self.label_203.setObjectName("label_203")
        self.formLayout_55.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_203)
        self.verticalLayout_103 = QtWidgets.QVBoxLayout()
        self.verticalLayout_103.setObjectName("verticalLayout_103")
        self.thinNever_8 = QtWidgets.QRadioButton(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_8.sizePolicy().hasHeightForWidth())
        self.thinNever_8.setSizePolicy(sizePolicy)
        self.thinNever_8.setObjectName("thinNever_8")
        self.verticalLayout_103.addWidget(self.thinNever_8)
        self.horizontalLayout_160 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_160.setSpacing(8)
        self.horizontalLayout_160.setObjectName("horizontalLayout_160")
        self.thinInterval_8 = QtWidgets.QRadioButton(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_8.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_8.setSizePolicy(sizePolicy)
        self.thinInterval_8.setObjectName("thinInterval_8")
        self.horizontalLayout_160.addWidget(self.thinInterval_8)
        self.thinFreq_8 = QtWidgets.QSpinBox(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_8.sizePolicy().hasHeightForWidth())
        self.thinFreq_8.setSizePolicy(sizePolicy)
        self.thinFreq_8.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_8.setMinimum(1)
        self.thinFreq_8.setSingleStep(10)
        self.thinFreq_8.setProperty("value", 1)
        self.thinFreq_8.setObjectName("thinFreq_8")
        self.horizontalLayout_160.addWidget(self.thinFreq_8)
        self.thinFreqLabel_8 = QtWidgets.QLabel(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_8.setSizePolicy(sizePolicy)
        self.thinFreqLabel_8.setObjectName("thinFreqLabel_8")
        self.horizontalLayout_160.addWidget(self.thinFreqLabel_8)
        self.verticalLayout_103.addLayout(self.horizontalLayout_160)
        self.thinCustom_8 = QtWidgets.QRadioButton(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_8.sizePolicy().hasHeightForWidth())
        self.thinCustom_8.setSizePolicy(sizePolicy)
        self.thinCustom_8.setObjectName("thinCustom_8")
        self.verticalLayout_103.addWidget(self.thinCustom_8)
        self.formLayout_55.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_103
        )
        self.thinCustomText_8 = QtWidgets.QLabel(self.widget_43)
        self.thinCustomText_8.setObjectName("thinCustomText_8")
        self.formLayout_55.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_8
        )
        self.thinCustomLine_8 = QtWidgets.QLineEdit(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_8.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_8.setSizePolicy(sizePolicy)
        self.thinCustomLine_8.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_8.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_8.setObjectName("thinCustomLine_8")
        self.formLayout_55.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_8
        )
        self.treesThinnedText_8 = QtWidgets.QLabel(self.widget_43)
        self.treesThinnedText_8.setObjectName("treesThinnedText_8")
        self.formLayout_55.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_8
        )
        self.horizontalLayout_161 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_161.setSpacing(8)
        self.horizontalLayout_161.setObjectName("horizontalLayout_161")
        self.treesThinned_8 = QtWidgets.QSpinBox(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_8.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_8.setSizePolicy(sizePolicy)
        self.treesThinned_8.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_8.setMaximum(100)
        self.treesThinned_8.setSingleStep(10)
        self.treesThinned_8.setProperty("value", 50)
        self.treesThinned_8.setObjectName("treesThinned_8")
        self.horizontalLayout_161.addWidget(self.treesThinned_8)
        self.treesThinnedLabel_8 = QtWidgets.QLabel(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_8.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_8.setObjectName("treesThinnedLabel_8")
        self.horizontalLayout_161.addWidget(self.treesThinnedLabel_8)
        self.formLayout_55.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_161
        )
        self.thinnedLeftText_8 = QtWidgets.QLabel(self.widget_43)
        self.thinnedLeftText_8.setObjectName("thinnedLeftText_8")
        self.formLayout_55.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_8
        )
        self.thinnedStemsLeftText_8 = QtWidgets.QLabel(self.widget_43)
        self.thinnedStemsLeftText_8.setObjectName("thinnedStemsLeftText_8")
        self.formLayout_55.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_8
        )
        self.horizontalLayout_162 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_162.setSpacing(8)
        self.horizontalLayout_162.setObjectName("horizontalLayout_162")
        self.thinnedStemsLeft_8 = QtWidgets.QSpinBox(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_8.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_8.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_8.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_8.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_8.setMaximum(100)
        self.thinnedStemsLeft_8.setSingleStep(10)
        self.thinnedStemsLeft_8.setObjectName("thinnedStemsLeft_8")
        self.horizontalLayout_162.addWidget(self.thinnedStemsLeft_8)
        self.thinnedStemsLeftLabel_8 = QtWidgets.QLabel(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_8.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_8.setObjectName("thinnedStemsLeftLabel_8")
        self.horizontalLayout_162.addWidget(self.thinnedStemsLeftLabel_8)
        self.formLayout_55.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_162
        )
        self.thinnedBranchesLeftText_8 = QtWidgets.QLabel(self.widget_43)
        self.thinnedBranchesLeftText_8.setObjectName("thinnedBranchesLeftText_8")
        self.formLayout_55.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_8
        )
        self.horizontalLayout_163 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_163.setSpacing(8)
        self.horizontalLayout_163.setObjectName("horizontalLayout_163")
        self.thinnedBranchesLeft_8 = QtWidgets.QSpinBox(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_8.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_8.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_8.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_8.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_8.setMaximum(100)
        self.thinnedBranchesLeft_8.setSingleStep(10)
        self.thinnedBranchesLeft_8.setObjectName("thinnedBranchesLeft_8")
        self.horizontalLayout_163.addWidget(self.thinnedBranchesLeft_8)
        self.thinnedBranchesLeftLabel_8 = QtWidgets.QLabel(self.widget_43)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_8.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_8.setObjectName("thinnedBranchesLeftLabel_8")
        self.horizontalLayout_163.addWidget(self.thinnedBranchesLeftLabel_8)
        self.formLayout_55.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_163
        )
        self.gridLayout_48.addLayout(self.formLayout_55, 1, 0, 1, 1)
        self.verticalLayout_104.addWidget(self.widget_43)
        self.stackedWidget.addWidget(self.treeThinPage_8)
        self.treeMortPage_8 = QtWidgets.QWidget()
        self.treeMortPage_8.setObjectName("treeMortPage_8")
        self.verticalLayout_105 = QtWidgets.QVBoxLayout(self.treeMortPage_8)
        self.verticalLayout_105.setObjectName("verticalLayout_105")
        self.widget_44 = QtWidgets.QWidget(self.treeMortPage_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_44.sizePolicy().hasHeightForWidth())
        self.widget_44.setSizePolicy(sizePolicy)
        self.widget_44.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_44.setObjectName("widget_44")
        self.gridLayout_49 = QtWidgets.QGridLayout(self.widget_44)
        self.gridLayout_49.setObjectName("gridLayout_49")
        self.label_204 = QtWidgets.QLabel(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_204.sizePolicy().hasHeightForWidth())
        self.label_204.setSizePolicy(sizePolicy)
        self.label_204.setMinimumSize(QtCore.QSize(0, 30))
        self.label_204.setObjectName("label_204")
        self.gridLayout_49.addWidget(self.label_204, 0, 0, 1, 1)
        self.formLayout_56 = QtWidgets.QFormLayout()
        self.formLayout_56.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_56.setVerticalSpacing(15)
        self.formLayout_56.setObjectName("formLayout_56")
        spacerItem48 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_56.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem48)
        self.label_205 = QtWidgets.QLabel(self.widget_44)
        self.label_205.setObjectName("label_205")
        self.formLayout_56.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_205)
        self.horizontalLayout_164 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_164.setSpacing(8)
        self.horizontalLayout_164.setObjectName("horizontalLayout_164")
        self.treesDead_8 = QtWidgets.QSpinBox(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_8.sizePolicy().hasHeightForWidth())
        self.treesDead_8.setSizePolicy(sizePolicy)
        self.treesDead_8.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_8.setMaximum(100)
        self.treesDead_8.setSingleStep(10)
        self.treesDead_8.setProperty("value", 1)
        self.treesDead_8.setObjectName("treesDead_8")
        self.horizontalLayout_164.addWidget(self.treesDead_8)
        self.label_206 = QtWidgets.QLabel(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_206.sizePolicy().hasHeightForWidth())
        self.label_206.setSizePolicy(sizePolicy)
        self.label_206.setObjectName("label_206")
        self.horizontalLayout_164.addWidget(self.label_206)
        self.formLayout_56.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_164
        )
        self.deadLeftText_8 = QtWidgets.QLabel(self.widget_44)
        self.deadLeftText_8.setObjectName("deadLeftText_8")
        self.formLayout_56.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_8
        )
        self.deadStemsLeftText_8 = QtWidgets.QLabel(self.widget_44)
        self.deadStemsLeftText_8.setObjectName("deadStemsLeftText_8")
        self.formLayout_56.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_8
        )
        self.horizontalLayout_165 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_165.setSpacing(8)
        self.horizontalLayout_165.setObjectName("horizontalLayout_165")
        self.deadStemsLeft_8 = QtWidgets.QSpinBox(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_8.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_8.setSizePolicy(sizePolicy)
        self.deadStemsLeft_8.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_8.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_8.setMaximum(100)
        self.deadStemsLeft_8.setSingleStep(10)
        self.deadStemsLeft_8.setObjectName("deadStemsLeft_8")
        self.horizontalLayout_165.addWidget(self.deadStemsLeft_8)
        self.deadStemsLeftLabel_8 = QtWidgets.QLabel(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_8.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_8.setObjectName("deadStemsLeftLabel_8")
        self.horizontalLayout_165.addWidget(self.deadStemsLeftLabel_8)
        self.formLayout_56.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_165
        )
        self.deadBranchesLeftText_8 = QtWidgets.QLabel(self.widget_44)
        self.deadBranchesLeftText_8.setObjectName("deadBranchesLeftText_8")
        self.formLayout_56.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_8
        )
        self.horizontalLayout_166 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_166.setSpacing(8)
        self.horizontalLayout_166.setObjectName("horizontalLayout_166")
        self.deadBranchesLeft_8 = QtWidgets.QSpinBox(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_8.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_8.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_8.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_8.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_8.setMaximum(100)
        self.deadBranchesLeft_8.setSingleStep(10)
        self.deadBranchesLeft_8.setObjectName("deadBranchesLeft_8")
        self.horizontalLayout_166.addWidget(self.deadBranchesLeft_8)
        self.deadBranchesLeftLabel_8 = QtWidgets.QLabel(self.widget_44)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_8.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_8.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_8.setObjectName("deadBranchesLeftLabel_8")
        self.horizontalLayout_166.addWidget(self.deadBranchesLeftLabel_8)
        self.formLayout_56.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_166
        )
        self.gridLayout_49.addLayout(self.formLayout_56, 1, 0, 1, 1)
        self.verticalLayout_105.addWidget(self.widget_44)
        self.stackedWidget.addWidget(self.treeMortPage_8)
        self.treeGrowthPage_8 = QtWidgets.QWidget()
        self.treeGrowthPage_8.setObjectName("treeGrowthPage_8")
        self.verticalLayout_107 = QtWidgets.QVBoxLayout(self.treeGrowthPage_8)
        self.verticalLayout_107.setObjectName("verticalLayout_107")
        self.widget_45 = QtWidgets.QWidget(self.treeGrowthPage_8)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_45.sizePolicy().hasHeightForWidth())
        self.widget_45.setSizePolicy(sizePolicy)
        self.widget_45.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_45.setObjectName("widget_45")
        self.gridLayout_50 = QtWidgets.QGridLayout(self.widget_45)
        self.gridLayout_50.setObjectName("gridLayout_50")
        self.formLayout_57 = QtWidgets.QFormLayout()
        self.formLayout_57.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_57.setVerticalSpacing(15)
        self.formLayout_57.setObjectName("formLayout_57")
        spacerItem49 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_57.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem49)
        self.verticalLayout_106 = QtWidgets.QVBoxLayout()
        self.verticalLayout_106.setSpacing(8)
        self.verticalLayout_106.setObjectName("verticalLayout_106")
        self.growthCsvButton_8 = QtWidgets.QRadioButton(self.widget_45)
        self.growthCsvButton_8.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_8.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_8.setSizePolicy(sizePolicy)
        self.growthCsvButton_8.setObjectName("growthCsvButton_8")
        self.verticalLayout_106.addWidget(self.growthCsvButton_8)
        self.growthCustomButton_8 = QtWidgets.QRadioButton(self.widget_45)
        self.growthCustomButton_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_8.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_8.setSizePolicy(sizePolicy)
        self.growthCustomButton_8.setObjectName("growthCustomButton_8")
        self.verticalLayout_106.addWidget(self.growthCustomButton_8)
        self.formLayout_57.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_106
        )
        self.growthCsvLabel_8 = QtWidgets.QLabel(self.widget_45)
        self.growthCsvLabel_8.setEnabled(True)
        self.growthCsvLabel_8.setObjectName("growthCsvLabel_8")
        self.formLayout_57.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_8
        )
        self.horizontalLayout_167 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_167.setSpacing(8)
        self.horizontalLayout_167.setObjectName("horizontalLayout_167")
        self.growthCsvBrowseBox_8 = QtWidgets.QLineEdit(self.widget_45)
        self.growthCsvBrowseBox_8.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_8.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_8.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_8.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_8.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_8.setMaxLength(1000)
        self.growthCsvBrowseBox_8.setObjectName("growthCsvBrowseBox_8")
        self.horizontalLayout_167.addWidget(self.growthCsvBrowseBox_8)
        self.growthCsvBrowseButton_8 = QtWidgets.QPushButton(self.widget_45)
        self.growthCsvBrowseButton_8.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_8.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_8.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_8.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_8.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_8.setObjectName("growthCsvBrowseButton_8")
        self.horizontalLayout_167.addWidget(self.growthCsvBrowseButton_8)
        self.formLayout_57.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_167
        )
        self.label_207 = QtWidgets.QLabel(self.widget_45)
        self.label_207.setObjectName("label_207")
        self.formLayout_57.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_207)
        self.allom_8 = QtWidgets.QComboBox(self.widget_45)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_8.sizePolicy().hasHeightForWidth())
        self.allom_8.setSizePolicy(sizePolicy)
        self.allom_8.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_8.setObjectName("allom_8")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.setItemText(1, "")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.setItemText(7, "")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.allom_8.setItemText(12, "")
        self.allom_8.addItem("")
        self.allom_8.addItem("")
        self.formLayout_57.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_8)
        self.gridLayout_50.addLayout(self.formLayout_57, 1, 0, 1, 1)
        self.label_208 = QtWidgets.QLabel(self.widget_45)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_208.sizePolicy().hasHeightForWidth())
        self.label_208.setSizePolicy(sizePolicy)
        self.label_208.setMinimumSize(QtCore.QSize(0, 30))
        self.label_208.setObjectName("label_208")
        self.gridLayout_50.addWidget(self.label_208, 0, 0, 1, 1)
        self.verticalLayout_107.addWidget(self.widget_45)
        self.stackedWidget.addWidget(self.treeGrowthPage_8)
        self.treeMgmtPage_9 = QtWidgets.QWidget()
        self.treeMgmtPage_9.setObjectName("treeMgmtPage_9")
        self.verticalLayout_109 = QtWidgets.QVBoxLayout(self.treeMgmtPage_9)
        self.verticalLayout_109.setObjectName("verticalLayout_109")
        self.widget_46 = QtWidgets.QWidget(self.treeMgmtPage_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_46.sizePolicy().hasHeightForWidth())
        self.widget_46.setSizePolicy(sizePolicy)
        self.widget_46.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_46.setObjectName("widget_46")
        self.gridLayout_51 = QtWidgets.QGridLayout(self.widget_46)
        self.gridLayout_51.setObjectName("gridLayout_51")
        self.label_209 = QtWidgets.QLabel(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_209.sizePolicy().hasHeightForWidth())
        self.label_209.setSizePolicy(sizePolicy)
        self.label_209.setMinimumSize(QtCore.QSize(0, 30))
        self.label_209.setObjectName("label_209")
        self.gridLayout_51.addWidget(self.label_209, 0, 0, 1, 1)
        self.formLayout_58 = QtWidgets.QFormLayout()
        self.formLayout_58.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_58.setVerticalSpacing(15)
        self.formLayout_58.setObjectName("formLayout_58")
        spacerItem50 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_58.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem50)
        self.label_210 = QtWidgets.QLabel(self.widget_46)
        self.label_210.setObjectName("label_210")
        self.formLayout_58.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_210)
        self.treeType_9 = QtWidgets.QComboBox(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeType_9.sizePolicy().hasHeightForWidth())
        self.treeType_9.setSizePolicy(sizePolicy)
        self.treeType_9.setMinimumSize(QtCore.QSize(200, 0))
        self.treeType_9.setObjectName("treeType_9")
        self.treeType_9.addItem("")
        self.treeType_9.addItem("")
        self.treeType_9.setItemText(1, "")
        self.treeType_9.addItem("")
        self.treeType_9.addItem("")
        self.treeType_9.addItem("")
        self.treeType_9.setItemText(4, "")
        self.treeType_9.addItem("")
        self.treeType_9.addItem("")
        self.formLayout_58.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.treeType_9
        )
        self.treeInputType_9 = QtWidgets.QLabel(self.widget_46)
        self.treeInputType_9.setEnabled(False)
        self.treeInputType_9.setObjectName("treeInputType_9")
        self.formLayout_58.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treeInputType_9
        )
        self.verticalLayout_108 = QtWidgets.QVBoxLayout()
        self.verticalLayout_108.setSpacing(8)
        self.verticalLayout_108.setObjectName("verticalLayout_108")
        self.treeCsvButton_9 = QtWidgets.QRadioButton(self.widget_46)
        self.treeCsvButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvButton_9.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvButton_9.setSizePolicy(sizePolicy)
        self.treeCsvButton_9.setObjectName("treeCsvButton_9")
        self.verticalLayout_108.addWidget(self.treeCsvButton_9)
        self.treeCustomButton_9 = QtWidgets.QRadioButton(self.widget_46)
        self.treeCustomButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCustomButton_9.sizePolicy().hasHeightForWidth()
        )
        self.treeCustomButton_9.setSizePolicy(sizePolicy)
        self.treeCustomButton_9.setObjectName("treeCustomButton_9")
        self.verticalLayout_108.addWidget(self.treeCustomButton_9)
        self.formLayout_58.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_108
        )
        self.treeCsvLabel_9 = QtWidgets.QLabel(self.widget_46)
        self.treeCsvLabel_9.setEnabled(False)
        self.treeCsvLabel_9.setObjectName("treeCsvLabel_9")
        self.formLayout_58.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.treeCsvLabel_9
        )
        self.horizontalLayout_168 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_168.setSpacing(8)
        self.horizontalLayout_168.setObjectName("horizontalLayout_168")
        self.treeCsvBrowseBox_9 = QtWidgets.QLineEdit(self.widget_46)
        self.treeCsvBrowseBox_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseBox_9.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseBox_9.setSizePolicy(sizePolicy)
        self.treeCsvBrowseBox_9.setMinimumSize(QtCore.QSize(230, 0))
        self.treeCsvBrowseBox_9.setMaximumSize(QtCore.QSize(230, 16777215))
        self.treeCsvBrowseBox_9.setMaxLength(1000)
        self.treeCsvBrowseBox_9.setObjectName("treeCsvBrowseBox_9")
        self.horizontalLayout_168.addWidget(self.treeCsvBrowseBox_9)
        self.treeCsvBrowseButton_9 = QtWidgets.QPushButton(self.widget_46)
        self.treeCsvBrowseButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treeCsvBrowseButton_9.sizePolicy().hasHeightForWidth()
        )
        self.treeCsvBrowseButton_9.setSizePolicy(sizePolicy)
        self.treeCsvBrowseButton_9.setMinimumSize(QtCore.QSize(85, 0))
        self.treeCsvBrowseButton_9.setMaximumSize(QtCore.QSize(85, 16777215))
        self.treeCsvBrowseButton_9.setObjectName("treeCsvBrowseButton_9")
        self.horizontalLayout_168.addWidget(self.treeCsvBrowseButton_9)
        self.formLayout_58.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_168
        )
        self.label_211 = QtWidgets.QLabel(self.widget_46)
        self.label_211.setObjectName("label_211")
        self.formLayout_58.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_211)
        self.horizontalLayout_169 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_169.setSpacing(8)
        self.horizontalLayout_169.setObjectName("horizontalLayout_169")
        self.standDens_9 = QtWidgets.QSpinBox(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standDens_9.sizePolicy().hasHeightForWidth())
        self.standDens_9.setSizePolicy(sizePolicy)
        self.standDens_9.setMinimumSize(QtCore.QSize(60, 0))
        self.standDens_9.setMaximum(999)
        self.standDens_9.setSingleStep(50)
        self.standDens_9.setProperty("value", 200)
        self.standDens_9.setObjectName("standDens_9")
        self.horizontalLayout_169.addWidget(self.standDens_9)
        self.label_212 = QtWidgets.QLabel(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_212.sizePolicy().hasHeightForWidth())
        self.label_212.setSizePolicy(sizePolicy)
        self.label_212.setObjectName("label_212")
        self.horizontalLayout_169.addWidget(self.label_212)
        self.formLayout_58.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_169
        )
        self.label_213 = QtWidgets.QLabel(self.widget_46)
        self.label_213.setObjectName("label_213")
        self.formLayout_58.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_213)
        self.horizontalLayout_170 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_170.setSpacing(8)
        self.horizontalLayout_170.setObjectName("horizontalLayout_170")
        self.label_214 = QtWidgets.QLabel(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_214.sizePolicy().hasHeightForWidth())
        self.label_214.setSizePolicy(sizePolicy)
        self.label_214.setObjectName("label_214")
        self.horizontalLayout_170.addWidget(self.label_214)
        self.yearPlanted_9 = QtWidgets.QSpinBox(self.widget_46)
        self.yearPlanted_9.setMinimumSize(QtCore.QSize(60, 0))
        self.yearPlanted_9.setMaximumSize(QtCore.QSize(60, 16777215))
        self.yearPlanted_9.setObjectName("yearPlanted_9")
        self.horizontalLayout_170.addWidget(self.yearPlanted_9)
        self.formLayout_58.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_170
        )
        self.label_15 = QtWidgets.QLabel(self.widget_46)
        self.label_15.setObjectName("label_15")
        self.formLayout_58.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.speciesName_9 = QtWidgets.QLineEdit(self.widget_46)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.speciesName_9.sizePolicy().hasHeightForWidth()
        )
        self.speciesName_9.setSizePolicy(sizePolicy)
        self.speciesName_9.setMinimumSize(QtCore.QSize(270, 0))
        self.speciesName_9.setMaximumSize(QtCore.QSize(270, 16777215))
        self.speciesName_9.setObjectName("speciesName_9")
        self.formLayout_58.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.speciesName_9
        )
        self.gridLayout_51.addLayout(self.formLayout_58, 1, 0, 1, 1)
        self.verticalLayout_109.addWidget(self.widget_46)
        self.stackedWidget.addWidget(self.treeMgmtPage_9)
        self.treeThinPage_9 = QtWidgets.QWidget()
        self.treeThinPage_9.setObjectName("treeThinPage_9")
        self.verticalLayout_111 = QtWidgets.QVBoxLayout(self.treeThinPage_9)
        self.verticalLayout_111.setObjectName("verticalLayout_111")
        self.widget_47 = QtWidgets.QWidget(self.treeThinPage_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_47.sizePolicy().hasHeightForWidth())
        self.widget_47.setSizePolicy(sizePolicy)
        self.widget_47.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_47.setObjectName("widget_47")
        self.gridLayout_52 = QtWidgets.QGridLayout(self.widget_47)
        self.gridLayout_52.setObjectName("gridLayout_52")
        self.label_215 = QtWidgets.QLabel(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_215.sizePolicy().hasHeightForWidth())
        self.label_215.setSizePolicy(sizePolicy)
        self.label_215.setMinimumSize(QtCore.QSize(0, 30))
        self.label_215.setObjectName("label_215")
        self.gridLayout_52.addWidget(self.label_215, 0, 0, 1, 1)
        self.formLayout_59 = QtWidgets.QFormLayout()
        self.formLayout_59.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_59.setVerticalSpacing(15)
        self.formLayout_59.setObjectName("formLayout_59")
        spacerItem51 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_59.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem51)
        self.label_216 = QtWidgets.QLabel(self.widget_47)
        self.label_216.setObjectName("label_216")
        self.formLayout_59.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_216)
        self.verticalLayout_110 = QtWidgets.QVBoxLayout()
        self.verticalLayout_110.setObjectName("verticalLayout_110")
        self.thinNever_9 = QtWidgets.QRadioButton(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinNever_9.sizePolicy().hasHeightForWidth())
        self.thinNever_9.setSizePolicy(sizePolicy)
        self.thinNever_9.setObjectName("thinNever_9")
        self.verticalLayout_110.addWidget(self.thinNever_9)
        self.horizontalLayout_171 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_171.setSpacing(8)
        self.horizontalLayout_171.setObjectName("horizontalLayout_171")
        self.thinInterval_9 = QtWidgets.QRadioButton(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinInterval_9.sizePolicy().hasHeightForWidth()
        )
        self.thinInterval_9.setSizePolicy(sizePolicy)
        self.thinInterval_9.setObjectName("thinInterval_9")
        self.horizontalLayout_171.addWidget(self.thinInterval_9)
        self.thinFreq_9 = QtWidgets.QSpinBox(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinFreq_9.sizePolicy().hasHeightForWidth())
        self.thinFreq_9.setSizePolicy(sizePolicy)
        self.thinFreq_9.setMinimumSize(QtCore.QSize(60, 0))
        self.thinFreq_9.setMinimum(1)
        self.thinFreq_9.setSingleStep(10)
        self.thinFreq_9.setProperty("value", 1)
        self.thinFreq_9.setObjectName("thinFreq_9")
        self.horizontalLayout_171.addWidget(self.thinFreq_9)
        self.thinFreqLabel_9 = QtWidgets.QLabel(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinFreqLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.thinFreqLabel_9.setSizePolicy(sizePolicy)
        self.thinFreqLabel_9.setObjectName("thinFreqLabel_9")
        self.horizontalLayout_171.addWidget(self.thinFreqLabel_9)
        self.verticalLayout_110.addLayout(self.horizontalLayout_171)
        self.thinCustom_9 = QtWidgets.QRadioButton(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thinCustom_9.sizePolicy().hasHeightForWidth())
        self.thinCustom_9.setSizePolicy(sizePolicy)
        self.thinCustom_9.setObjectName("thinCustom_9")
        self.verticalLayout_110.addWidget(self.thinCustom_9)
        self.formLayout_59.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_110
        )
        self.thinCustomText_9 = QtWidgets.QLabel(self.widget_47)
        self.thinCustomText_9.setObjectName("thinCustomText_9")
        self.formLayout_59.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.thinCustomText_9
        )
        self.thinCustomLine_9 = QtWidgets.QLineEdit(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinCustomLine_9.sizePolicy().hasHeightForWidth()
        )
        self.thinCustomLine_9.setSizePolicy(sizePolicy)
        self.thinCustomLine_9.setMinimumSize(QtCore.QSize(270, 0))
        self.thinCustomLine_9.setMaximumSize(QtCore.QSize(270, 16777215))
        self.thinCustomLine_9.setObjectName("thinCustomLine_9")
        self.formLayout_59.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.thinCustomLine_9
        )
        self.treesThinnedText_9 = QtWidgets.QLabel(self.widget_47)
        self.treesThinnedText_9.setObjectName("treesThinnedText_9")
        self.formLayout_59.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.treesThinnedText_9
        )
        self.horizontalLayout_172 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_172.setSpacing(8)
        self.horizontalLayout_172.setObjectName("horizontalLayout_172")
        self.treesThinned_9 = QtWidgets.QSpinBox(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinned_9.sizePolicy().hasHeightForWidth()
        )
        self.treesThinned_9.setSizePolicy(sizePolicy)
        self.treesThinned_9.setMinimumSize(QtCore.QSize(60, 0))
        self.treesThinned_9.setMaximum(100)
        self.treesThinned_9.setSingleStep(10)
        self.treesThinned_9.setProperty("value", 50)
        self.treesThinned_9.setObjectName("treesThinned_9")
        self.horizontalLayout_172.addWidget(self.treesThinned_9)
        self.treesThinnedLabel_9 = QtWidgets.QLabel(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.treesThinnedLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.treesThinnedLabel_9.setSizePolicy(sizePolicy)
        self.treesThinnedLabel_9.setObjectName("treesThinnedLabel_9")
        self.horizontalLayout_172.addWidget(self.treesThinnedLabel_9)
        self.formLayout_59.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_172
        )
        self.thinnedLeftText_9 = QtWidgets.QLabel(self.widget_47)
        self.thinnedLeftText_9.setObjectName("thinnedLeftText_9")
        self.formLayout_59.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.thinnedLeftText_9
        )
        self.thinnedStemsLeftText_9 = QtWidgets.QLabel(self.widget_47)
        self.thinnedStemsLeftText_9.setObjectName("thinnedStemsLeftText_9")
        self.formLayout_59.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.thinnedStemsLeftText_9
        )
        self.horizontalLayout_173 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_173.setSpacing(8)
        self.horizontalLayout_173.setObjectName("horizontalLayout_173")
        self.thinnedStemsLeft_9 = QtWidgets.QSpinBox(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeft_9.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeft_9.setSizePolicy(sizePolicy)
        self.thinnedStemsLeft_9.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedStemsLeft_9.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedStemsLeft_9.setMaximum(100)
        self.thinnedStemsLeft_9.setSingleStep(10)
        self.thinnedStemsLeft_9.setObjectName("thinnedStemsLeft_9")
        self.horizontalLayout_173.addWidget(self.thinnedStemsLeft_9)
        self.thinnedStemsLeftLabel_9 = QtWidgets.QLabel(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedStemsLeftLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.thinnedStemsLeftLabel_9.setSizePolicy(sizePolicy)
        self.thinnedStemsLeftLabel_9.setObjectName("thinnedStemsLeftLabel_9")
        self.horizontalLayout_173.addWidget(self.thinnedStemsLeftLabel_9)
        self.formLayout_59.setLayout(
            6, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_173
        )
        self.thinnedBranchesLeftText_9 = QtWidgets.QLabel(self.widget_47)
        self.thinnedBranchesLeftText_9.setObjectName("thinnedBranchesLeftText_9")
        self.formLayout_59.setWidget(
            7, QtWidgets.QFormLayout.LabelRole, self.thinnedBranchesLeftText_9
        )
        self.horizontalLayout_174 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_174.setSpacing(8)
        self.horizontalLayout_174.setObjectName("horizontalLayout_174")
        self.thinnedBranchesLeft_9 = QtWidgets.QSpinBox(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeft_9.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeft_9.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeft_9.setMinimumSize(QtCore.QSize(60, 0))
        self.thinnedBranchesLeft_9.setMaximumSize(QtCore.QSize(60, 16777215))
        self.thinnedBranchesLeft_9.setMaximum(100)
        self.thinnedBranchesLeft_9.setSingleStep(10)
        self.thinnedBranchesLeft_9.setObjectName("thinnedBranchesLeft_9")
        self.horizontalLayout_174.addWidget(self.thinnedBranchesLeft_9)
        self.thinnedBranchesLeftLabel_9 = QtWidgets.QLabel(self.widget_47)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.thinnedBranchesLeftLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.thinnedBranchesLeftLabel_9.setSizePolicy(sizePolicy)
        self.thinnedBranchesLeftLabel_9.setObjectName("thinnedBranchesLeftLabel_9")
        self.horizontalLayout_174.addWidget(self.thinnedBranchesLeftLabel_9)
        self.formLayout_59.setLayout(
            7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_174
        )
        self.gridLayout_52.addLayout(self.formLayout_59, 1, 0, 1, 1)
        self.verticalLayout_111.addWidget(self.widget_47)
        self.stackedWidget.addWidget(self.treeThinPage_9)
        self.treeMortPage_9 = QtWidgets.QWidget()
        self.treeMortPage_9.setObjectName("treeMortPage_9")
        self.verticalLayout_112 = QtWidgets.QVBoxLayout(self.treeMortPage_9)
        self.verticalLayout_112.setObjectName("verticalLayout_112")
        self.widget_48 = QtWidgets.QWidget(self.treeMortPage_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_48.sizePolicy().hasHeightForWidth())
        self.widget_48.setSizePolicy(sizePolicy)
        self.widget_48.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_48.setObjectName("widget_48")
        self.gridLayout_53 = QtWidgets.QGridLayout(self.widget_48)
        self.gridLayout_53.setObjectName("gridLayout_53")
        self.label_217 = QtWidgets.QLabel(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_217.sizePolicy().hasHeightForWidth())
        self.label_217.setSizePolicy(sizePolicy)
        self.label_217.setMinimumSize(QtCore.QSize(0, 30))
        self.label_217.setObjectName("label_217")
        self.gridLayout_53.addWidget(self.label_217, 0, 0, 1, 1)
        self.formLayout_60 = QtWidgets.QFormLayout()
        self.formLayout_60.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_60.setVerticalSpacing(15)
        self.formLayout_60.setObjectName("formLayout_60")
        spacerItem52 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_60.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem52)
        self.label_218 = QtWidgets.QLabel(self.widget_48)
        self.label_218.setObjectName("label_218")
        self.formLayout_60.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_218)
        self.horizontalLayout_175 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_175.setSpacing(8)
        self.horizontalLayout_175.setObjectName("horizontalLayout_175")
        self.treesDead_9 = QtWidgets.QSpinBox(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treesDead_9.sizePolicy().hasHeightForWidth())
        self.treesDead_9.setSizePolicy(sizePolicy)
        self.treesDead_9.setMinimumSize(QtCore.QSize(60, 0))
        self.treesDead_9.setMaximum(100)
        self.treesDead_9.setSingleStep(10)
        self.treesDead_9.setProperty("value", 1)
        self.treesDead_9.setObjectName("treesDead_9")
        self.horizontalLayout_175.addWidget(self.treesDead_9)
        self.label_219 = QtWidgets.QLabel(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_219.sizePolicy().hasHeightForWidth())
        self.label_219.setSizePolicy(sizePolicy)
        self.label_219.setObjectName("label_219")
        self.horizontalLayout_175.addWidget(self.label_219)
        self.formLayout_60.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_175
        )
        self.deadLeftText_9 = QtWidgets.QLabel(self.widget_48)
        self.deadLeftText_9.setObjectName("deadLeftText_9")
        self.formLayout_60.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.deadLeftText_9
        )
        self.deadStemsLeftText_9 = QtWidgets.QLabel(self.widget_48)
        self.deadStemsLeftText_9.setObjectName("deadStemsLeftText_9")
        self.formLayout_60.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.deadStemsLeftText_9
        )
        self.horizontalLayout_176 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_176.setSpacing(8)
        self.horizontalLayout_176.setObjectName("horizontalLayout_176")
        self.deadStemsLeft_9 = QtWidgets.QSpinBox(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeft_9.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeft_9.setSizePolicy(sizePolicy)
        self.deadStemsLeft_9.setMinimumSize(QtCore.QSize(60, 0))
        self.deadStemsLeft_9.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadStemsLeft_9.setMaximum(100)
        self.deadStemsLeft_9.setSingleStep(10)
        self.deadStemsLeft_9.setObjectName("deadStemsLeft_9")
        self.horizontalLayout_176.addWidget(self.deadStemsLeft_9)
        self.deadStemsLeftLabel_9 = QtWidgets.QLabel(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadStemsLeftLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.deadStemsLeftLabel_9.setSizePolicy(sizePolicy)
        self.deadStemsLeftLabel_9.setObjectName("deadStemsLeftLabel_9")
        self.horizontalLayout_176.addWidget(self.deadStemsLeftLabel_9)
        self.formLayout_60.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_176
        )
        self.deadBranchesLeftText_9 = QtWidgets.QLabel(self.widget_48)
        self.deadBranchesLeftText_9.setObjectName("deadBranchesLeftText_9")
        self.formLayout_60.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.deadBranchesLeftText_9
        )
        self.horizontalLayout_177 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_177.setSpacing(8)
        self.horizontalLayout_177.setObjectName("horizontalLayout_177")
        self.deadBranchesLeft_9 = QtWidgets.QSpinBox(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeft_9.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeft_9.setSizePolicy(sizePolicy)
        self.deadBranchesLeft_9.setMinimumSize(QtCore.QSize(60, 0))
        self.deadBranchesLeft_9.setMaximumSize(QtCore.QSize(60, 16777215))
        self.deadBranchesLeft_9.setMaximum(100)
        self.deadBranchesLeft_9.setSingleStep(10)
        self.deadBranchesLeft_9.setObjectName("deadBranchesLeft_9")
        self.horizontalLayout_177.addWidget(self.deadBranchesLeft_9)
        self.deadBranchesLeftLabel_9 = QtWidgets.QLabel(self.widget_48)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.deadBranchesLeftLabel_9.sizePolicy().hasHeightForWidth()
        )
        self.deadBranchesLeftLabel_9.setSizePolicy(sizePolicy)
        self.deadBranchesLeftLabel_9.setObjectName("deadBranchesLeftLabel_9")
        self.horizontalLayout_177.addWidget(self.deadBranchesLeftLabel_9)
        self.formLayout_60.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_177
        )
        self.gridLayout_53.addLayout(self.formLayout_60, 1, 0, 1, 1)
        self.verticalLayout_112.addWidget(self.widget_48)
        self.stackedWidget.addWidget(self.treeMortPage_9)
        self.treeGrowthPage_9 = QtWidgets.QWidget()
        self.treeGrowthPage_9.setObjectName("treeGrowthPage_9")
        self.verticalLayout_114 = QtWidgets.QVBoxLayout(self.treeGrowthPage_9)
        self.verticalLayout_114.setObjectName("verticalLayout_114")
        self.widget_49 = QtWidgets.QWidget(self.treeGrowthPage_9)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_49.sizePolicy().hasHeightForWidth())
        self.widget_49.setSizePolicy(sizePolicy)
        self.widget_49.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_49.setObjectName("widget_49")
        self.gridLayout_54 = QtWidgets.QGridLayout(self.widget_49)
        self.gridLayout_54.setObjectName("gridLayout_54")
        self.formLayout_61 = QtWidgets.QFormLayout()
        self.formLayout_61.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_61.setVerticalSpacing(15)
        self.formLayout_61.setObjectName("formLayout_61")
        spacerItem53 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_61.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem53)
        self.verticalLayout_113 = QtWidgets.QVBoxLayout()
        self.verticalLayout_113.setSpacing(8)
        self.verticalLayout_113.setObjectName("verticalLayout_113")
        self.growthCsvButton_9 = QtWidgets.QRadioButton(self.widget_49)
        self.growthCsvButton_9.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvButton_9.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvButton_9.setSizePolicy(sizePolicy)
        self.growthCsvButton_9.setObjectName("growthCsvButton_9")
        self.verticalLayout_113.addWidget(self.growthCsvButton_9)
        self.growthCustomButton_9 = QtWidgets.QRadioButton(self.widget_49)
        self.growthCustomButton_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCustomButton_9.sizePolicy().hasHeightForWidth()
        )
        self.growthCustomButton_9.setSizePolicy(sizePolicy)
        self.growthCustomButton_9.setObjectName("growthCustomButton_9")
        self.verticalLayout_113.addWidget(self.growthCustomButton_9)
        self.formLayout_61.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_113
        )
        self.growthCsvLabel_9 = QtWidgets.QLabel(self.widget_49)
        self.growthCsvLabel_9.setEnabled(True)
        self.growthCsvLabel_9.setObjectName("growthCsvLabel_9")
        self.formLayout_61.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.growthCsvLabel_9
        )
        self.horizontalLayout_178 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_178.setSpacing(8)
        self.horizontalLayout_178.setObjectName("horizontalLayout_178")
        self.growthCsvBrowseBox_9 = QtWidgets.QLineEdit(self.widget_49)
        self.growthCsvBrowseBox_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseBox_9.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseBox_9.setSizePolicy(sizePolicy)
        self.growthCsvBrowseBox_9.setMinimumSize(QtCore.QSize(230, 0))
        self.growthCsvBrowseBox_9.setMaximumSize(QtCore.QSize(230, 16777215))
        self.growthCsvBrowseBox_9.setMaxLength(1000)
        self.growthCsvBrowseBox_9.setObjectName("growthCsvBrowseBox_9")
        self.horizontalLayout_178.addWidget(self.growthCsvBrowseBox_9)
        self.growthCsvBrowseButton_9 = QtWidgets.QPushButton(self.widget_49)
        self.growthCsvBrowseButton_9.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.growthCsvBrowseButton_9.sizePolicy().hasHeightForWidth()
        )
        self.growthCsvBrowseButton_9.setSizePolicy(sizePolicy)
        self.growthCsvBrowseButton_9.setMinimumSize(QtCore.QSize(85, 0))
        self.growthCsvBrowseButton_9.setMaximumSize(QtCore.QSize(85, 16777215))
        self.growthCsvBrowseButton_9.setObjectName("growthCsvBrowseButton_9")
        self.horizontalLayout_178.addWidget(self.growthCsvBrowseButton_9)
        self.formLayout_61.setLayout(
            3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_178
        )
        self.label_220 = QtWidgets.QLabel(self.widget_49)
        self.label_220.setObjectName("label_220")
        self.formLayout_61.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_220)
        self.allom_9 = QtWidgets.QComboBox(self.widget_49)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allom_9.sizePolicy().hasHeightForWidth())
        self.allom_9.setSizePolicy(sizePolicy)
        self.allom_9.setMinimumSize(QtCore.QSize(200, 0))
        self.allom_9.setObjectName("allom_9")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.setItemText(1, "")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.setItemText(7, "")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.allom_9.setItemText(12, "")
        self.allom_9.addItem("")
        self.allom_9.addItem("")
        self.formLayout_61.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.allom_9)
        self.gridLayout_54.addLayout(self.formLayout_61, 1, 0, 1, 1)
        self.label_221 = QtWidgets.QLabel(self.widget_49)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_221.sizePolicy().hasHeightForWidth())
        self.label_221.setSizePolicy(sizePolicy)
        self.label_221.setMinimumSize(QtCore.QSize(0, 30))
        self.label_221.setObjectName("label_221")
        self.gridLayout_54.addWidget(self.label_221, 0, 0, 1, 1)
        self.verticalLayout_114.addWidget(self.widget_49)
        self.stackedWidget.addWidget(self.treeGrowthPage_9)
        self.litterPage_0 = QtWidgets.QWidget()
        self.litterPage_0.setObjectName("litterPage_0")
        self.verticalLayout_65 = QtWidgets.QVBoxLayout(self.litterPage_0)
        self.verticalLayout_65.setObjectName("verticalLayout_65")
        self.widget_10 = QtWidgets.QWidget(self.litterPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_10.sizePolicy().hasHeightForWidth())
        self.widget_10.setSizePolicy(sizePolicy)
        self.widget_10.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_10.setObjectName("widget_10")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_10)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_57 = QtWidgets.QLabel(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_57.sizePolicy().hasHeightForWidth())
        self.label_57.setSizePolicy(sizePolicy)
        self.label_57.setMinimumSize(QtCore.QSize(0, 30))
        self.label_57.setObjectName("label_57")
        self.verticalLayout_6.addWidget(self.label_57)
        self.formLayout_14 = QtWidgets.QFormLayout()
        self.formLayout_14.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_14.setVerticalSpacing(15)
        self.formLayout_14.setObjectName("formLayout_14")
        spacerItem54 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_14.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem54)
        self.label_28 = QtWidgets.QLabel(self.widget_10)
        self.label_28.setObjectName("label_28")
        self.formLayout_14.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_28)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_35 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_35.setSpacing(8)
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.litterInterval_0 = QtWidgets.QRadioButton(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInterval_0.sizePolicy().hasHeightForWidth()
        )
        self.litterInterval_0.setSizePolicy(sizePolicy)
        self.litterInterval_0.setObjectName("litterInterval_0")
        self.horizontalLayout_35.addWidget(self.litterInterval_0)
        self.litterFreq_0 = QtWidgets.QSpinBox(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.litterFreq_0.sizePolicy().hasHeightForWidth())
        self.litterFreq_0.setSizePolicy(sizePolicy)
        self.litterFreq_0.setMinimumSize(QtCore.QSize(60, 0))
        self.litterFreq_0.setMinimum(1)
        self.litterFreq_0.setSingleStep(10)
        self.litterFreq_0.setProperty("value", 1)
        self.litterFreq_0.setObjectName("litterFreq_0")
        self.horizontalLayout_35.addWidget(self.litterFreq_0)
        self.litterFreqLabel_0 = QtWidgets.QLabel(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterFreqLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.litterFreqLabel_0.setSizePolicy(sizePolicy)
        self.litterFreqLabel_0.setObjectName("litterFreqLabel_0")
        self.horizontalLayout_35.addWidget(self.litterFreqLabel_0)
        self.verticalLayout_4.addLayout(self.horizontalLayout_35)
        self.litterCustom_0 = QtWidgets.QRadioButton(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterCustom_0.sizePolicy().hasHeightForWidth()
        )
        self.litterCustom_0.setSizePolicy(sizePolicy)
        self.litterCustom_0.setObjectName("litterCustom_0")
        self.verticalLayout_4.addWidget(self.litterCustom_0)
        self.formLayout_14.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_4
        )
        self.litterCustomText_0 = QtWidgets.QLabel(self.widget_10)
        self.litterCustomText_0.setObjectName("litterCustomText_0")
        self.formLayout_14.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.litterCustomText_0
        )
        self.litterCustomLine_0 = QtWidgets.QLineEdit(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterCustomLine_0.sizePolicy().hasHeightForWidth()
        )
        self.litterCustomLine_0.setSizePolicy(sizePolicy)
        self.litterCustomLine_0.setMinimumSize(QtCore.QSize(270, 0))
        self.litterCustomLine_0.setMaximumSize(QtCore.QSize(270, 16777215))
        self.litterCustomLine_0.setObjectName("litterCustomLine_0")
        self.formLayout_14.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.litterCustomLine_0
        )
        self.litterInputText_0 = QtWidgets.QLabel(self.widget_10)
        self.litterInputText_0.setObjectName("litterInputText_0")
        self.formLayout_14.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.litterInputText_0
        )
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setSpacing(8)
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.litterInput_0 = QtWidgets.QDoubleSpinBox(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInput_0.sizePolicy().hasHeightForWidth()
        )
        self.litterInput_0.setSizePolicy(sizePolicy)
        self.litterInput_0.setMinimumSize(QtCore.QSize(80, 0))
        self.litterInput_0.setDecimals(1)
        self.litterInput_0.setMaximum(20000.0)
        self.litterInput_0.setSingleStep(10.0)
        self.litterInput_0.setProperty("value", 0.0)
        self.litterInput_0.setObjectName("litterInput_0")
        self.horizontalLayout_26.addWidget(self.litterInput_0)
        self.litterInputLabel_0 = QtWidgets.QLabel(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInputLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.litterInputLabel_0.setSizePolicy(sizePolicy)
        self.litterInputLabel_0.setObjectName("litterInputLabel_0")
        self.horizontalLayout_26.addWidget(self.litterInputLabel_0)
        self.formLayout_14.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_26
        )
        self.verticalLayout_6.addLayout(self.formLayout_14)
        self.verticalLayout_65.addWidget(self.widget_10)
        self.stackedWidget.addWidget(self.litterPage_0)
        self.litterPage_1 = QtWidgets.QWidget()
        self.litterPage_1.setObjectName("litterPage_1")
        self.verticalLayout_40 = QtWidgets.QVBoxLayout(self.litterPage_1)
        self.verticalLayout_40.setObjectName("verticalLayout_40")
        self.widget_13 = QtWidgets.QWidget(self.litterPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_13.sizePolicy().hasHeightForWidth())
        self.widget_13.setSizePolicy(sizePolicy)
        self.widget_13.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_13.setObjectName("widget_13")
        self.verticalLayout_38 = QtWidgets.QVBoxLayout(self.widget_13)
        self.verticalLayout_38.setObjectName("verticalLayout_38")
        self.label_88 = QtWidgets.QLabel(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_88.sizePolicy().hasHeightForWidth())
        self.label_88.setSizePolicy(sizePolicy)
        self.label_88.setMinimumSize(QtCore.QSize(0, 30))
        self.label_88.setObjectName("label_88")
        self.verticalLayout_38.addWidget(self.label_88)
        self.formLayout_20 = QtWidgets.QFormLayout()
        self.formLayout_20.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_20.setVerticalSpacing(15)
        self.formLayout_20.setObjectName("formLayout_20")
        spacerItem55 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_20.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem55)
        self.label_31 = QtWidgets.QLabel(self.widget_13)
        self.label_31.setObjectName("label_31")
        self.formLayout_20.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_31)
        self.verticalLayout_39 = QtWidgets.QVBoxLayout()
        self.verticalLayout_39.setObjectName("verticalLayout_39")
        self.horizontalLayout_59 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_59.setSpacing(8)
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.litterInterval_1 = QtWidgets.QRadioButton(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInterval_1.sizePolicy().hasHeightForWidth()
        )
        self.litterInterval_1.setSizePolicy(sizePolicy)
        self.litterInterval_1.setObjectName("litterInterval_1")
        self.horizontalLayout_59.addWidget(self.litterInterval_1)
        self.litterFreq_1 = QtWidgets.QSpinBox(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.litterFreq_1.sizePolicy().hasHeightForWidth())
        self.litterFreq_1.setSizePolicy(sizePolicy)
        self.litterFreq_1.setMinimumSize(QtCore.QSize(60, 0))
        self.litterFreq_1.setMinimum(1)
        self.litterFreq_1.setSingleStep(10)
        self.litterFreq_1.setProperty("value", 1)
        self.litterFreq_1.setObjectName("litterFreq_1")
        self.horizontalLayout_59.addWidget(self.litterFreq_1)
        self.litterFreqLabel_1 = QtWidgets.QLabel(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterFreqLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.litterFreqLabel_1.setSizePolicy(sizePolicy)
        self.litterFreqLabel_1.setObjectName("litterFreqLabel_1")
        self.horizontalLayout_59.addWidget(self.litterFreqLabel_1)
        self.verticalLayout_39.addLayout(self.horizontalLayout_59)
        self.litterCustom_1 = QtWidgets.QRadioButton(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterCustom_1.sizePolicy().hasHeightForWidth()
        )
        self.litterCustom_1.setSizePolicy(sizePolicy)
        self.litterCustom_1.setObjectName("litterCustom_1")
        self.verticalLayout_39.addWidget(self.litterCustom_1)
        self.formLayout_20.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_39
        )
        self.litterCustomText_1 = QtWidgets.QLabel(self.widget_13)
        self.litterCustomText_1.setObjectName("litterCustomText_1")
        self.formLayout_20.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.litterCustomText_1
        )
        self.litterCustomLine_1 = QtWidgets.QLineEdit(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterCustomLine_1.sizePolicy().hasHeightForWidth()
        )
        self.litterCustomLine_1.setSizePolicy(sizePolicy)
        self.litterCustomLine_1.setMinimumSize(QtCore.QSize(270, 0))
        self.litterCustomLine_1.setMaximumSize(QtCore.QSize(270, 16777215))
        self.litterCustomLine_1.setObjectName("litterCustomLine_1")
        self.formLayout_20.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.litterCustomLine_1
        )
        self.litterInputText_1 = QtWidgets.QLabel(self.widget_13)
        self.litterInputText_1.setObjectName("litterInputText_1")
        self.formLayout_20.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.litterInputText_1
        )
        self.horizontalLayout_58 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_58.setSpacing(8)
        self.horizontalLayout_58.setObjectName("horizontalLayout_58")
        self.litterInput_1 = QtWidgets.QDoubleSpinBox(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInput_1.sizePolicy().hasHeightForWidth()
        )
        self.litterInput_1.setSizePolicy(sizePolicy)
        self.litterInput_1.setMinimumSize(QtCore.QSize(80, 0))
        self.litterInput_1.setDecimals(1)
        self.litterInput_1.setMaximum(20.0)
        self.litterInput_1.setSingleStep(0.5)
        self.litterInput_1.setProperty("value", 0.0)
        self.litterInput_1.setObjectName("litterInput_1")
        self.horizontalLayout_58.addWidget(self.litterInput_1)
        self.litterInputLabel_1 = QtWidgets.QLabel(self.widget_13)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.litterInputLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.litterInputLabel_1.setSizePolicy(sizePolicy)
        self.litterInputLabel_1.setObjectName("litterInputLabel_1")
        self.horizontalLayout_58.addWidget(self.litterInputLabel_1)
        self.formLayout_20.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_58
        )
        self.verticalLayout_38.addLayout(self.formLayout_20)
        self.verticalLayout_40.addWidget(self.widget_13)
        self.stackedWidget.addWidget(self.litterPage_1)
        self.fertPage_0 = QtWidgets.QWidget()
        self.fertPage_0.setObjectName("fertPage_0")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.fertPage_0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.widget_11 = QtWidgets.QWidget(self.fertPage_0)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_11.sizePolicy().hasHeightForWidth())
        self.widget_11.setSizePolicy(sizePolicy)
        self.widget_11.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_11.setObjectName("widget_11")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_11)
        self.gridLayout.setObjectName("gridLayout")
        self.label_59 = QtWidgets.QLabel(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_59.sizePolicy().hasHeightForWidth())
        self.label_59.setSizePolicy(sizePolicy)
        self.label_59.setMinimumSize(QtCore.QSize(0, 30))
        self.label_59.setObjectName("label_59")
        self.gridLayout.addWidget(self.label_59, 0, 0, 1, 1)
        self.formLayout_15 = QtWidgets.QFormLayout()
        self.formLayout_15.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_15.setVerticalSpacing(15)
        self.formLayout_15.setObjectName("formLayout_15")
        spacerItem56 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_15.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem56)
        self.label_29 = QtWidgets.QLabel(self.widget_11)
        self.label_29.setObjectName("label_29")
        self.formLayout_15.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_39 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_39.setSpacing(8)
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.fertInterval_0 = QtWidgets.QRadioButton(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertInterval_0.sizePolicy().hasHeightForWidth()
        )
        self.fertInterval_0.setSizePolicy(sizePolicy)
        self.fertInterval_0.setObjectName("fertInterval_0")
        self.horizontalLayout_39.addWidget(self.fertInterval_0)
        self.fertFreq_0 = QtWidgets.QSpinBox(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertFreq_0.sizePolicy().hasHeightForWidth())
        self.fertFreq_0.setSizePolicy(sizePolicy)
        self.fertFreq_0.setMinimumSize(QtCore.QSize(60, 0))
        self.fertFreq_0.setMinimum(1)
        self.fertFreq_0.setSingleStep(10)
        self.fertFreq_0.setProperty("value", 1)
        self.fertFreq_0.setObjectName("fertFreq_0")
        self.horizontalLayout_39.addWidget(self.fertFreq_0)
        self.fertFreqLabel_0 = QtWidgets.QLabel(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertFreqLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.fertFreqLabel_0.setSizePolicy(sizePolicy)
        self.fertFreqLabel_0.setObjectName("fertFreqLabel_0")
        self.horizontalLayout_39.addWidget(self.fertFreqLabel_0)
        self.verticalLayout_11.addLayout(self.horizontalLayout_39)
        self.fertCustom_0 = QtWidgets.QRadioButton(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertCustom_0.sizePolicy().hasHeightForWidth())
        self.fertCustom_0.setSizePolicy(sizePolicy)
        self.fertCustom_0.setObjectName("fertCustom_0")
        self.verticalLayout_11.addWidget(self.fertCustom_0)
        self.formLayout_15.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_11
        )
        self.fertCustomText_0 = QtWidgets.QLabel(self.widget_11)
        self.fertCustomText_0.setObjectName("fertCustomText_0")
        self.formLayout_15.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.fertCustomText_0
        )
        self.fertCustomLine_0 = QtWidgets.QLineEdit(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertCustomLine_0.sizePolicy().hasHeightForWidth()
        )
        self.fertCustomLine_0.setSizePolicy(sizePolicy)
        self.fertCustomLine_0.setMinimumSize(QtCore.QSize(270, 0))
        self.fertCustomLine_0.setMaximumSize(QtCore.QSize(270, 16777215))
        self.fertCustomLine_0.setObjectName("fertCustomLine_0")
        self.formLayout_15.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.fertCustomLine_0
        )
        self.fertInputText_0 = QtWidgets.QLabel(self.widget_11)
        self.fertInputText_0.setObjectName("fertInputText_0")
        self.formLayout_15.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.fertInputText_0
        )
        self.horizontalLayout_37 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_37.setSpacing(8)
        self.horizontalLayout_37.setObjectName("horizontalLayout_37")
        self.fertInput_0 = QtWidgets.QDoubleSpinBox(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertInput_0.sizePolicy().hasHeightForWidth())
        self.fertInput_0.setSizePolicy(sizePolicy)
        self.fertInput_0.setMinimumSize(QtCore.QSize(80, 0))
        self.fertInput_0.setDecimals(1)
        self.fertInput_0.setMaximum(20000.0)
        self.fertInput_0.setSingleStep(50.0)
        self.fertInput_0.setProperty("value", 0.0)
        self.fertInput_0.setObjectName("fertInput_0")
        self.horizontalLayout_37.addWidget(self.fertInput_0)
        self.fertInputLabel_0 = QtWidgets.QLabel(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertInputLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.fertInputLabel_0.setSizePolicy(sizePolicy)
        self.fertInputLabel_0.setObjectName("fertInputLabel_0")
        self.horizontalLayout_37.addWidget(self.fertInputLabel_0)
        self.formLayout_15.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_37
        )
        self.fertNitrogenText_0 = QtWidgets.QLabel(self.widget_11)
        self.fertNitrogenText_0.setObjectName("fertNitrogenText_0")
        self.formLayout_15.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.fertNitrogenText_0
        )
        self.horizontalLayout_40 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_40.setSpacing(8)
        self.horizontalLayout_40.setObjectName("horizontalLayout_40")
        self.fertNitrogen_0 = QtWidgets.QSpinBox(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertNitrogen_0.sizePolicy().hasHeightForWidth()
        )
        self.fertNitrogen_0.setSizePolicy(sizePolicy)
        self.fertNitrogen_0.setMinimumSize(QtCore.QSize(60, 0))
        self.fertNitrogen_0.setMaximum(100)
        self.fertNitrogen_0.setProperty("value", 10)
        self.fertNitrogen_0.setObjectName("fertNitrogen_0")
        self.horizontalLayout_40.addWidget(self.fertNitrogen_0)
        self.fertNitrogenLabel_0 = QtWidgets.QLabel(self.widget_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertNitrogenLabel_0.sizePolicy().hasHeightForWidth()
        )
        self.fertNitrogenLabel_0.setSizePolicy(sizePolicy)
        self.fertNitrogenLabel_0.setObjectName("fertNitrogenLabel_0")
        self.horizontalLayout_40.addWidget(self.fertNitrogenLabel_0)
        self.formLayout_15.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_40
        )
        self.gridLayout.addLayout(self.formLayout_15, 1, 0, 1, 1)
        self.verticalLayout_12.addWidget(self.widget_11)
        self.stackedWidget.addWidget(self.fertPage_0)
        self.fertPage_1 = QtWidgets.QWidget()
        self.fertPage_1.setObjectName("fertPage_1")
        self.verticalLayout_33 = QtWidgets.QVBoxLayout(self.fertPage_1)
        self.verticalLayout_33.setObjectName("verticalLayout_33")
        self.widget_12 = QtWidgets.QWidget(self.fertPage_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_12.sizePolicy().hasHeightForWidth())
        self.widget_12.setSizePolicy(sizePolicy)
        self.widget_12.setMinimumSize(QtCore.QSize(620, 200))
        self.widget_12.setObjectName("widget_12")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.widget_12)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_87 = QtWidgets.QLabel(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_87.sizePolicy().hasHeightForWidth())
        self.label_87.setSizePolicy(sizePolicy)
        self.label_87.setMinimumSize(QtCore.QSize(0, 30))
        self.label_87.setObjectName("label_87")
        self.gridLayout_12.addWidget(self.label_87, 0, 0, 1, 1)
        self.formLayout_19 = QtWidgets.QFormLayout()
        self.formLayout_19.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        self.formLayout_19.setVerticalSpacing(15)
        self.formLayout_19.setObjectName("formLayout_19")
        spacerItem57 = QtWidgets.QSpacerItem(
            250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.formLayout_19.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem57)
        self.label_30 = QtWidgets.QLabel(self.widget_12)
        self.label_30.setObjectName("label_30")
        self.formLayout_19.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.verticalLayout_31 = QtWidgets.QVBoxLayout()
        self.verticalLayout_31.setObjectName("verticalLayout_31")
        self.horizontalLayout_56 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_56.setSpacing(8)
        self.horizontalLayout_56.setObjectName("horizontalLayout_56")
        self.fertInterval_1 = QtWidgets.QRadioButton(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertInterval_1.sizePolicy().hasHeightForWidth()
        )
        self.fertInterval_1.setSizePolicy(sizePolicy)
        self.fertInterval_1.setObjectName("fertInterval_1")
        self.horizontalLayout_56.addWidget(self.fertInterval_1)
        self.fertFreq_1 = QtWidgets.QSpinBox(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertFreq_1.sizePolicy().hasHeightForWidth())
        self.fertFreq_1.setSizePolicy(sizePolicy)
        self.fertFreq_1.setMinimumSize(QtCore.QSize(60, 0))
        self.fertFreq_1.setMinimum(1)
        self.fertFreq_1.setSingleStep(10)
        self.fertFreq_1.setProperty("value", 1)
        self.fertFreq_1.setObjectName("fertFreq_1")
        self.horizontalLayout_56.addWidget(self.fertFreq_1)
        self.fertFreqLabel_1 = QtWidgets.QLabel(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertFreqLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.fertFreqLabel_1.setSizePolicy(sizePolicy)
        self.fertFreqLabel_1.setObjectName("fertFreqLabel_1")
        self.horizontalLayout_56.addWidget(self.fertFreqLabel_1)
        self.verticalLayout_31.addLayout(self.horizontalLayout_56)
        self.fertCustom_1 = QtWidgets.QRadioButton(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertCustom_1.sizePolicy().hasHeightForWidth())
        self.fertCustom_1.setSizePolicy(sizePolicy)
        self.fertCustom_1.setObjectName("fertCustom_1")
        self.verticalLayout_31.addWidget(self.fertCustom_1)
        self.formLayout_19.setLayout(
            2, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_31
        )
        self.fertCustomText_1 = QtWidgets.QLabel(self.widget_12)
        self.fertCustomText_1.setObjectName("fertCustomText_1")
        self.formLayout_19.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.fertCustomText_1
        )
        self.fertCustomLine_1 = QtWidgets.QLineEdit(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertCustomLine_1.sizePolicy().hasHeightForWidth()
        )
        self.fertCustomLine_1.setSizePolicy(sizePolicy)
        self.fertCustomLine_1.setMinimumSize(QtCore.QSize(270, 0))
        self.fertCustomLine_1.setMaximumSize(QtCore.QSize(270, 16777215))
        self.fertCustomLine_1.setObjectName("fertCustomLine_1")
        self.formLayout_19.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.fertCustomLine_1
        )
        self.fertInputText_1 = QtWidgets.QLabel(self.widget_12)
        self.fertInputText_1.setObjectName("fertInputText_1")
        self.formLayout_19.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.fertInputText_1
        )
        self.horizontalLayout_55 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_55.setSpacing(8)
        self.horizontalLayout_55.setObjectName("horizontalLayout_55")
        self.fertInput_1 = QtWidgets.QDoubleSpinBox(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(80)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fertInput_1.sizePolicy().hasHeightForWidth())
        self.fertInput_1.setSizePolicy(sizePolicy)
        self.fertInput_1.setMinimumSize(QtCore.QSize(80, 0))
        self.fertInput_1.setDecimals(1)
        self.fertInput_1.setMaximum(20000.0)
        self.fertInput_1.setSingleStep(50.0)
        self.fertInput_1.setProperty("value", 0.0)
        self.fertInput_1.setObjectName("fertInput_1")
        self.horizontalLayout_55.addWidget(self.fertInput_1)
        self.fertInputLabel_1 = QtWidgets.QLabel(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertInputLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.fertInputLabel_1.setSizePolicy(sizePolicy)
        self.fertInputLabel_1.setObjectName("fertInputLabel_1")
        self.horizontalLayout_55.addWidget(self.fertInputLabel_1)
        self.formLayout_19.setLayout(
            4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_55
        )
        self.fertNitrogenText_1 = QtWidgets.QLabel(self.widget_12)
        self.fertNitrogenText_1.setObjectName("fertNitrogenText_1")
        self.formLayout_19.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.fertNitrogenText_1
        )
        self.horizontalLayout_57 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_57.setSpacing(8)
        self.horizontalLayout_57.setObjectName("horizontalLayout_57")
        self.fertNitrogen_1 = QtWidgets.QSpinBox(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(60)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertNitrogen_1.sizePolicy().hasHeightForWidth()
        )
        self.fertNitrogen_1.setSizePolicy(sizePolicy)
        self.fertNitrogen_1.setMinimumSize(QtCore.QSize(60, 0))
        self.fertNitrogen_1.setMaximum(100)
        self.fertNitrogen_1.setProperty("value", 10)
        self.fertNitrogen_1.setObjectName("fertNitrogen_1")
        self.horizontalLayout_57.addWidget(self.fertNitrogen_1)
        self.fertNitrogenLabel_1 = QtWidgets.QLabel(self.widget_12)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fertNitrogenLabel_1.sizePolicy().hasHeightForWidth()
        )
        self.fertNitrogenLabel_1.setSizePolicy(sizePolicy)
        self.fertNitrogenLabel_1.setObjectName("fertNitrogenLabel_1")
        self.horizontalLayout_57.addWidget(self.fertNitrogenLabel_1)
        self.formLayout_19.setLayout(
            5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_57
        )
        self.gridLayout_12.addLayout(self.formLayout_19, 1, 0, 1, 1)
        self.verticalLayout_33.addWidget(self.widget_12)
        self.stackedWidget.addWidget(self.fertPage_1)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_116 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_116.setObjectName("verticalLayout_116")
        self.widget_50 = QtWidgets.QWidget(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_50.sizePolicy().hasHeightForWidth())
        self.widget_50.setSizePolicy(sizePolicy)
        self.widget_50.setMinimumSize(QtCore.QSize(620, 175))
        self.widget_50.setObjectName("widget_50")
        self.verticalLayout_115 = QtWidgets.QVBoxLayout(self.widget_50)
        self.verticalLayout_115.setObjectName("verticalLayout_115")
        self.end_title_label = QtWidgets.QLabel(self.widget_50)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.end_title_label.sizePolicy().hasHeightForWidth()
        )
        self.end_title_label.setSizePolicy(sizePolicy)
        self.end_title_label.setObjectName("end_title_label")
        self.verticalLayout_115.addWidget(self.end_title_label)
        self.label_223 = QtWidgets.QLabel(self.widget_50)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_223.sizePolicy().hasHeightForWidth())
        self.label_223.setSizePolicy(sizePolicy)
        self.label_223.setObjectName("label_223")
        self.verticalLayout_115.addWidget(self.label_223)
        self.verticalLayout_116.addWidget(self.widget_50)
        spacerItem58 = QtWidgets.QSpacerItem(
            20, 247, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_116.addItem(spacerItem58)
        self.stackedWidget.addWidget(self.page)
        self.verticalLayout_8.addWidget(self.stackedWidget)
        self.verticalLayout.addLayout(self.verticalLayout_8)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.helpButton = QtWidgets.QDialogButtonBox(project)
        self.helpButton.setStandardButtons(QtWidgets.QDialogButtonBox.Help)
        self.helpButton.setObjectName("helpButton")
        self.horizontalLayout_8.addWidget(self.helpButton)
        self.backButton = QtWidgets.QPushButton(project)
        self.backButton.setCheckable(False)
        self.backButton.setObjectName("backButton")
        self.horizontalLayout_8.addWidget(self.backButton)
        self.nextButton = QtWidgets.QPushButton(project)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout_8.addWidget(self.nextButton)
        self.verticalLayout.addLayout(self.horizontalLayout_8)

        self.retranslateUi(project)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(project)

    def retranslateUi(self, project):
        project.setWindowTitle(
            QtWidgets.QApplication.translate("project", "Baseline Scenario", None)
        )
        self.title_label.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:18pt; font-weight:600;">NEW BASELINE SCENARIO</span></p></body></html>',
                None,
            )
        )
        self.subtitle_label.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">The following screens will prompt you to enter </span></p>\n'
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">information about the farming activities that make </span></p>\n'
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">up the baseline scenario</span></p></body></html>',
                None,
            )
        )
        self.label_11.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">OVERVIEW</span></p></body></html>',
                None,
            )
        )
        self.cropNumLabel.setText(
            QtWidgets.QApplication.translate(
                "project", "Number of crop species planted", None
            )
        )
        self.treeNumLabel.setText(
            QtWidgets.QApplication.translate(
                "project", "Number of tree species planted", None
            )
        )
        self.litterNumLabel.setText(
            QtWidgets.QApplication.translate(
                "project", "Number of types of external\n" "organic inputs added", None
            )
        )
        self.label.setText(
            QtWidgets.QApplication.translate(
                "project", "Number of types of synthetic\n" "fertiliser added", None
            )
        )
        self.label_10.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Soil Data</span></p></body></html>',
                None,
            )
        )
        self.soilFromHWSD.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Use SHAMBA default data for\n" "this location (recommended)",
                None,
            )
        )
        self.soilFromCsv.setText(
            QtWidgets.QApplication.translate("project", "Load soil data", None)
        )
        self.soilFromCustom.setText(
            QtWidgets.QApplication.translate(
                "project", "Enter soil data directly", None
            )
        )
        self.soilCsvLabel.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load", None)
        )
        self.soilCsvBrowseButton.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.soilCy0Label.setText(
            QtWidgets.QApplication.translate(
                "project", "Soil organic carbon at start of project", None
            )
        )
        self.soilCy0Unit.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes carbon per hectare", None
            )
        )
        self.soilClayLabel.setText(
            QtWidgets.QApplication.translate(
                "project", "Soil clay content at start of project", None
            )
        )
        self.soilClayUnit.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_36.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Soil and Fire Management</span></p></body></html>',
                None,
            )
        )
        self.soilCoverLabel.setText(
            QtWidgets.QApplication.translate(
                "project", "Check the months when\n" "soil is covered:", None
            )
        )
        self.allCover.setText(
            QtWidgets.QApplication.translate("project", "Select all", None)
        )
        self.janCover.setText(
            QtWidgets.QApplication.translate("project", "January", None)
        )
        self.febCover.setText(
            QtWidgets.QApplication.translate("project", "February", None)
        )
        self.marCover.setText(
            QtWidgets.QApplication.translate("project", "March", None)
        )
        self.aprCover.setText(
            QtWidgets.QApplication.translate("project", "April", None)
        )
        self.mayCover.setText(QtWidgets.QApplication.translate("project", "May", None))
        self.junCover.setText(QtWidgets.QApplication.translate("project", "June", None))
        self.julCover.setText(QtWidgets.QApplication.translate("project", "July", None))
        self.augCover.setText(
            QtWidgets.QApplication.translate("project", "August", None)
        )
        self.sepCover.setText(
            QtWidgets.QApplication.translate("project", "September", None)
        )
        self.octCover.setText(
            QtWidgets.QApplication.translate("project", "October", None)
        )
        self.novCover.setText(
            QtWidgets.QApplication.translate("project", "November", None)
        )
        self.decCover.setText(
            QtWidgets.QApplication.translate("project", "December", None)
        )
        self.label_39.setText(
            QtWidgets.QApplication.translate("project", "Burning", None)
        )
        self.fireNever.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.fireInterval.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.fireFreqLabel.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.fireCustom.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.fireCustomText.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Enter the years when burning\n" "is expected (e.g. 2,7-9,18)",
                None,
            )
        )
        self.label_46.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 1</span></p></body></html>',
                None,
            )
        )
        self.label_45.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_0.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_0.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_0.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_0.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_0.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_0.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_0.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_0.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_0.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_0.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_0.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_0.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_0.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_0.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_0.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_0.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_0.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_0.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_0.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_0.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_0.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_0.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_0.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_0.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_0.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_0.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_0.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_0.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_0.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_0.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_0.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_0.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_0.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_0.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_44.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_47.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_48.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_49.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_0.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_0.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_2.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_60.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 2</span></p></body></html>',
                None,
            )
        )
        self.label_61.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_1.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_1.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_1.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_1.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_1.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_1.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_1.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_1.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_1.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_1.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_1.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_1.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_1.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_1.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_1.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_1.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_1.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_1.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_1.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_1.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_1.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_1.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_1.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_1.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_1.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_1.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_1.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_1.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_1.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_1.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_1.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_1.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_1.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_1.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_63.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_64.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_65.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_68.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_1.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_1.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_3.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_69.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 3</span></p></body></html>',
                None,
            )
        )
        self.label_70.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_2.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_2.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_2.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_2.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_2.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_2.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_2.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_2.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_2.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_2.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_2.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_2.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_2.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_2.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_2.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_2.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_2.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_2.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_2.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_2.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_2.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_2.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_2.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_2.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_2.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_2.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_2.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_2.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_2.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_2.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_2.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_2.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_2.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_2.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_71.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_72.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_73.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.cropBurnYes_2.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_2.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_4.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_74.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.label_75.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 4</span></p></body></html>',
                None,
            )
        )
        self.label_76.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_3.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_3.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_3.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_3.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_3.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_3.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_3.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_3.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_3.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_3.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_3.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_3.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_3.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_3.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_3.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_3.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_3.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_3.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_3.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_3.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_3.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_3.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_3.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_3.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_3.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_3.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_3.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_3.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_3.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_3.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_3.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_3.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_3.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_3.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_77.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_78.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_79.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_80.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_3.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_3.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_5.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_81.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 5</span></p></body></html>',
                None,
            )
        )
        self.label_82.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_4.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_4.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_4.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_4.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_4.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_4.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_4.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_4.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_4.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_4.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_4.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_4.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_4.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_4.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_4.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_4.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_4.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_4.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_4.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_4.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_4.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_4.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_4.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_4.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_4.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_4.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_4.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_4.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_4.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_4.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_4.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_4.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_4.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_4.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_83.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_84.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_85.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_86.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_4.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_4.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_7.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_129.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 6</span></p></body></html>',
                None,
            )
        )
        self.label_130.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_5.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_5.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_5.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_5.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_5.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_5.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_5.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_5.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_5.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_5.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_5.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_5.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_5.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_5.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_5.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_5.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_5.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_5.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_5.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_5.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_5.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_5.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_5.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_5.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_5.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_5.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_5.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_5.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_5.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_5.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_5.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_5.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_5.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_5.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_131.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_132.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_133.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_134.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_5.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_5.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_8.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_135.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 7</span></p></body></html>',
                None,
            )
        )
        self.label_136.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_6.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_6.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_6.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_6.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_6.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_6.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_6.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_6.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_6.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_6.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_6.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_6.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_6.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_6.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_6.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_6.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_6.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_6.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_6.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_6.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_6.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_6.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_6.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_6.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_6.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_6.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_6.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_6.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_6.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_6.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_6.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_6.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_6.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_6.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_137.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_138.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_139.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_140.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_6.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_6.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_9.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_141.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 8</span></p></body></html>',
                None,
            )
        )
        self.label_142.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_7.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_7.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_7.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_7.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_7.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_7.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_7.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_7.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_7.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_7.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_7.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_7.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_7.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_7.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_7.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_7.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_7.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_7.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_7.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_7.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_7.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_7.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_7.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_7.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_7.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_7.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_7.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_7.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_7.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_7.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_7.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_7.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_7.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_7.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_143.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_144.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_145.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_146.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_7.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_7.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_12.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_147.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 9</span></p></body></html>',
                None,
            )
        )
        self.label_148.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_8.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_8.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_8.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_8.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_8.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_8.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_8.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_8.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_8.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_8.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_8.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_8.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_8.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_8.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_8.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_8.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_8.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_8.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_8.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_8.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_8.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_8.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_8.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_8.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_8.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_8.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_8.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_8.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_8.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_8.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_8.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_8.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_8.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_8.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_149.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_150.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_151.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_152.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_8.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_8.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_13.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_153.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Crop Management - Crop 10</span></p></body></html>',
                None,
            )
        )
        self.label_154.setText(
            QtWidgets.QApplication.translate("project", "Crop planted", None)
        )
        self.cropType_9.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.cropType_9.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Specific species--", None)
        )
        self.cropType_9.setItemText(
            3, QtWidgets.QApplication.translate("project", "maize", None)
        )
        self.cropType_9.setItemText(
            4, QtWidgets.QApplication.translate("project", "wheat", None)
        )
        self.cropType_9.setItemText(
            5, QtWidgets.QApplication.translate("project", "winter wheat", None)
        )
        self.cropType_9.setItemText(
            6, QtWidgets.QApplication.translate("project", "spring wheat", None)
        )
        self.cropType_9.setItemText(
            7, QtWidgets.QApplication.translate("project", "rice", None)
        )
        self.cropType_9.setItemText(
            8, QtWidgets.QApplication.translate("project", "barley", None)
        )
        self.cropType_9.setItemText(
            9, QtWidgets.QApplication.translate("project", "oats", None)
        )
        self.cropType_9.setItemText(
            10, QtWidgets.QApplication.translate("project", "millet", None)
        )
        self.cropType_9.setItemText(
            11, QtWidgets.QApplication.translate("project", "sorghum", None)
        )
        self.cropType_9.setItemText(
            12, QtWidgets.QApplication.translate("project", "rye", None)
        )
        self.cropType_9.setItemText(
            13, QtWidgets.QApplication.translate("project", "soyabean", None)
        )
        self.cropType_9.setItemText(
            14, QtWidgets.QApplication.translate("project", "dry bean", None)
        )
        self.cropType_9.setItemText(
            15, QtWidgets.QApplication.translate("project", "potato", None)
        )
        self.cropType_9.setItemText(
            16, QtWidgets.QApplication.translate("project", "peanut", None)
        )
        self.cropType_9.setItemText(
            17, QtWidgets.QApplication.translate("project", "alfalfa", None)
        )
        self.cropType_9.setItemText(
            18, QtWidgets.QApplication.translate("project", "non-legume hay", None)
        )
        self.cropType_9.setItemText(
            20, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.cropType_9.setItemText(
            21, QtWidgets.QApplication.translate("project", "grains", None)
        )
        self.cropType_9.setItemText(
            22, QtWidgets.QApplication.translate("project", "beans and pulses", None)
        )
        self.cropType_9.setItemText(
            23, QtWidgets.QApplication.translate("project", "tubers", None)
        )
        self.cropType_9.setItemText(
            24, QtWidgets.QApplication.translate("project", "root crops", None)
        )
        self.cropType_9.setItemText(
            25, QtWidgets.QApplication.translate("project", "N-fixing forages", None)
        )
        self.cropType_9.setItemText(
            26,
            QtWidgets.QApplication.translate("project", "non N-fixing forages", None),
        )
        self.cropType_9.setItemText(
            27, QtWidgets.QApplication.translate("project", "perennial grasses", None)
        )
        self.cropType_9.setItemText(
            28,
            QtWidgets.QApplication.translate("project", "grass-clover mixtures", None),
        )
        self.cropType_9.setItemText(
            30,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.cropType_9.setItemText(
            31, QtWidgets.QApplication.translate("project", "Other crop", None)
        )
        self.cropInputType_9.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced crop\n" "data (to come in v2)?", None
            )
        )
        self.cropCsvButton_9.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.cropCustomButton_9.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.cropCsvLabel_9.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.cropCsvBrowseButton_9.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_155.setText(
            QtWidgets.QApplication.translate("project", "Annual dry crop yield", None)
        )
        self.label_156.setText(
            QtWidgets.QApplication.translate(
                "project", "tonnes dry matter per hectare", None
            )
        )
        self.label_157.setText(
            QtWidgets.QApplication.translate(
                "project", "Percent of residues left\n" "in field after harvest", None
            )
        )
        self.label_158.setText(
            QtWidgets.QApplication.translate(
                "project", "% of total above-ground residues", None
            )
        )
        self.cropBurnYes_9.setText(
            QtWidgets.QApplication.translate("project", "Yes", None)
        )
        self.cropBurnNo_9.setText(
            QtWidgets.QApplication.translate("project", "No", None)
        )
        self.label_14.setText(
            QtWidgets.QApplication.translate(
                "project", "Removed crop residues burned?", None
            )
        )
        self.label_50.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 1 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_6.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_51.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_0.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_0.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_0.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_0.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_0.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_0.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_0.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_0.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_0.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_0.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_53.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_55.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_17.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_20.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_54.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 1 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_26.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_0.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_0.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_0.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_0.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_0.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_0.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_0.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_0.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_0.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_58.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 1 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_66.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_67.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_0.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_0.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_0.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_0.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_0.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_0.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_0.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_0.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_52.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_0.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_0.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_0.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_0.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_0.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_0.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_0.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_0.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_0.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_0.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_0.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_0.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_56.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 1 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_89.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 2 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_90.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_1.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_1.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_1.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_1.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_1.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_1.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_1.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_1.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_1.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_1.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_91.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_92.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_18.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_21.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_226.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_93.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 2 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_32.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_1.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_1.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_1.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_1.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_1.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_1.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_1.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_1.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_1.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_94.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 2 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_95.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_96.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_1.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_1.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_1.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_1.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_1.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_1.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_1.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_1.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_97.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_1.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_1.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_1.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_1.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_1.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_1.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_1.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_1.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_1.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_1.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_1.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_1.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_98.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 2 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_99.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 3 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_100.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_2.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_2.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_2.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_2.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_2.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_2.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_2.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_2.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_2.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_2.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_101.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_102.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_19.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_22.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_225.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_103.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 3 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_34.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_2.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_2.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_2.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_2.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_2.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_2.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_2.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_2.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_2.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_2.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_2.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_2.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_104.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 3 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_105.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_106.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_2.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_2.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_2.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_2.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_2.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_2.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_2.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_2.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_2.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_107.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_2.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_2.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_2.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_2.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_2.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_2.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_2.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_2.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_2.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_2.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_2.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_2.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_108.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 3 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_109.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 4 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_110.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_3.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_3.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_3.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_3.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_3.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_3.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_3.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_3.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_3.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_3.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_111.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_112.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_23.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_24.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_224.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_113.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 4 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_35.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_3.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_3.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_3.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_3.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_3.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_3.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_3.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_3.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_3.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_3.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_3.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_3.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_114.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 4 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_115.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_116.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_3.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_3.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_3.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_3.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_3.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_3.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_3.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_3.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_3.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_117.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_3.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_3.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_3.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_3.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_3.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_3.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_3.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_3.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_3.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_3.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_3.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_3.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_118.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 4 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_119.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 5 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_120.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_4.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_4.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_4.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_4.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_4.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_4.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_4.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_4.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_4.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_4.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_121.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_122.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_25.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_37.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_222.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_123.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 5 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_40.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_4.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_4.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_4.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_4.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_4.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_4.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_4.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_4.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_4.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_4.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_4.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_4.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_124.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 5 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_125.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_126.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_4.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_4.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_4.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_4.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_4.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_4.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_4.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_4.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_4.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_127.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_4.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_4.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_4.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_4.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_4.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_4.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_4.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_4.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_4.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_4.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_4.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_4.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_128.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 5 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_159.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 6 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_160.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_5.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_5.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_5.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_5.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_5.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_5.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_5.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_5.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_5.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_5.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_161.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_162.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_42.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_43.setText(QtWidgets.QApplication.translate("project", "year", None))
        self.label_62.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_163.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 6 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_164.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_5.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_5.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_5.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_5.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_5.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_5.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_5.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_5.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_5.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_5.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_5.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_5.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_165.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 6 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_166.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_167.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_5.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_5.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_5.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_5.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_5.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_5.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_5.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_5.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_5.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_168.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_5.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_5.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_5.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_5.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_5.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_5.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_5.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_5.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_5.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_5.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_5.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_5.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_169.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 6 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_170.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 7 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_171.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_6.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_6.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_6.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_6.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_6.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_6.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_6.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_6.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_6.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_6.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_172.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_173.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_174.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_175.setText(
            QtWidgets.QApplication.translate("project", "year", None)
        )
        self.label_38.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_176.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 7 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_177.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_6.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_6.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_6.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_6.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_6.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_6.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_6.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_6.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_6.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_6.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_6.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_6.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_178.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 7 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_179.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_180.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_6.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_6.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_6.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_6.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_6.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_6.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_6.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_6.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_6.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_181.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_6.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_6.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_6.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_6.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_6.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_6.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_6.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_6.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_6.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_6.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_6.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_6.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_182.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 7 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_183.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 8 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_184.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_7.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_7.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_7.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_7.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_7.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_7.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_7.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_7.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_7.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_7.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_185.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_186.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_187.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_188.setText(
            QtWidgets.QApplication.translate("project", "year", None)
        )
        self.label_27.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_189.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 8 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_190.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_7.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_7.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_7.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_7.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_7.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_7.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_7.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_7.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_7.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_7.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_7.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_7.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_191.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 8 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_192.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_193.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_7.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_7.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_7.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_7.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_7.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_7.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_7.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_7.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_7.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_194.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_7.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_7.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_7.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_7.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_7.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_7.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_7.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_7.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_7.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_7.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_7.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_7.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_195.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 8 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_196.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 9 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_197.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_8.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_8.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_8.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_8.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_8.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_8.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_8.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_8.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_8.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_8.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_199.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_200.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_201.setText(
            QtWidgets.QApplication.translate("project", "year", None)
        )
        self.label_198.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_16.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_202.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 9 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_203.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_8.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_8.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_8.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_8.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_8.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_8.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_8.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_8.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_8.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_8.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_8.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_8.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_204.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 9 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_205.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_206.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_8.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_8.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_8.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_8.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_8.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_8.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_8.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_8.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_8.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_207.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_8.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_8.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_8.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_8.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_8.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_8.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_8.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_8.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_8.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_8.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_8.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_8.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_208.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 9 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_209.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 10 - Planting</span></p></body></html>',
                None,
            )
        )
        self.label_210.setText(
            QtWidgets.QApplication.translate("project", "Species type", None)
        )
        self.treeType_9.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.treeType_9.setItemText(
            2, QtWidgets.QApplication.translate("project", "--Generic types--", None)
        )
        self.treeType_9.setItemText(
            3, QtWidgets.QApplication.translate("project", "default", None)
        )
        self.treeType_9.setItemText(
            5,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.treeType_9.setItemText(
            6, QtWidgets.QApplication.translate("project", "Other tree", None)
        )
        self.treeInputType_9.setText(
            QtWidgets.QApplication.translate(
                "project", "How to enter advanced tree\n" "data (to come in v2)?", None
            )
        )
        self.treeCsvButton_9.setText(
            QtWidgets.QApplication.translate("project", "Load from existing file", None)
        )
        self.treeCustomButton_9.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.treeCsvLabel_9.setText(
            QtWidgets.QApplication.translate("project", "Select a file to  load:", None)
        )
        self.treeCsvBrowseButton_9.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_211.setText(
            QtWidgets.QApplication.translate("project", "Planting density", None)
        )
        self.label_212.setText(
            QtWidgets.QApplication.translate("project", "per hectare", None)
        )
        self.label_213.setText(
            QtWidgets.QApplication.translate("project", "Year of planting", None)
        )
        self.label_214.setText(
            QtWidgets.QApplication.translate("project", "year", None)
        )
        self.label_15.setText(
            QtWidgets.QApplication.translate("project", "Species name", None)
        )
        self.label_215.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 10 - Thinning and Harvest</span></p></body></html>',
                None,
            )
        )
        self.label_216.setText(
            QtWidgets.QApplication.translate(
                "project", "When will trees be\n" "thinned or harvested", None
            )
        )
        self.thinNever_9.setText(
            QtWidgets.QApplication.translate("project", "Never", None)
        )
        self.thinInterval_9.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.thinFreqLabel_9.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.thinCustom_9.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.thinCustomText_9.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Please enter a list and/or range\n"
                "of years in which thinning occurs\n"
                "(e.g. 2,7-9,18):",
                None,
            )
        )
        self.treesThinnedText_9.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of trees felled during\n" "each harvest or thinning event",
                None,
            )
        )
        self.treesThinnedLabel_9.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedLeftText_9.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of felled tree stem and\n"
                "branch biomass left in the field",
                None,
            )
        )
        self.thinnedStemsLeftText_9.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.thinnedStemsLeftLabel_9.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.thinnedBranchesLeftText_9.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.thinnedBranchesLeftLabel_9.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.label_217.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 10 - Mortality</span></p></body></html>',
                None,
            )
        )
        self.label_218.setText(
            QtWidgets.QApplication.translate(
                "project", "Percentage of trees expected\n" "to die each year", None
            )
        )
        self.label_219.setText(QtWidgets.QApplication.translate("project", "%", None))
        self.deadLeftText_9.setText(
            QtWidgets.QApplication.translate(
                "project",
                "Percentage of stem and branch\n"
                "biomass from dead trees that will\n"
                "be left in the field",
                None,
            )
        )
        self.deadStemsLeftText_9.setText(
            QtWidgets.QApplication.translate("project", "        Stems", None)
        )
        self.deadStemsLeftLabel_9.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.deadBranchesLeftText_9.setText(
            QtWidgets.QApplication.translate("project", "        Branches", None)
        )
        self.deadBranchesLeftLabel_9.setText(
            QtWidgets.QApplication.translate("project", "%", None)
        )
        self.growthCsvButton_9.setText(
            QtWidgets.QApplication.translate(
                "project", "Load growth data from existing file", None
            )
        )
        self.growthCustomButton_9.setText(
            QtWidgets.QApplication.translate("project", "Enter custom data", None)
        )
        self.growthCsvLabel_9.setText(
            QtWidgets.QApplication.translate("project", "Select a file to load:", None)
        )
        self.growthCsvBrowseButton_9.setText(
            QtWidgets.QApplication.translate("project", "Browse", None)
        )
        self.label_220.setText(
            QtWidgets.QApplication.translate("project", "Select allometric model", None)
        )
        self.allom_9.setItemText(
            0, QtWidgets.QApplication.translate("project", "--Select--", None)
        )
        self.allom_9.setItemText(
            2, QtWidgets.QApplication.translate("project", "--General tropical--", None)
        )
        self.allom_9.setItemText(
            3, QtWidgets.QApplication.translate("project", "Ryan", None)
        )
        self.allom_9.setItemText(
            4, QtWidgets.QApplication.translate("project", "Chave dry", None)
        )
        self.allom_9.setItemText(
            5, QtWidgets.QApplication.translate("project", "Chave moist", None)
        )
        self.allom_9.setItemText(
            6, QtWidgets.QApplication.translate("project", "Chave wet", None)
        )
        self.allom_9.setItemText(
            8,
            QtWidgets.QApplication.translate(
                "project", "--Tumwebaze agroforestry trees--", None
            ),
        )
        self.allom_9.setItemText(
            9, QtWidgets.QApplication.translate("project", "Grevillea", None)
        )
        self.allom_9.setItemText(
            10, QtWidgets.QApplication.translate("project", "Maesopsis", None)
        )
        self.allom_9.setItemText(
            11, QtWidgets.QApplication.translate("project", "Markhamia", None)
        )
        self.allom_9.setItemText(
            13,
            QtWidgets.QApplication.translate(
                "project", "--Advanced (available in v2)--", None
            ),
        )
        self.allom_9.setItemText(
            14, QtWidgets.QApplication.translate("project", "Other allometric", None)
        )
        self.label_221.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Tree Species 10 - Growth</span></p></body></html>',
                None,
            )
        )
        self.label_57.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">External Organic Inputs - Type 1</span></p></body></html>',
                None,
            )
        )
        self.label_28.setText(
            QtWidgets.QApplication.translate(
                "project",
                "External Organic Inputs (EOIs)\n"
                "such as litter, mulch or manure\n"
                "added to the field",
                None,
            )
        )
        self.litterInterval_0.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.litterFreqLabel_0.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.litterCustom_0.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.litterCustomText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "List and/or range of years in which\n"
                "EOIs are added (e.g. 2,7-9,18):",
                None,
            )
        )
        self.litterInputText_0.setText(
            QtWidgets.QApplication.translate(
                "project", "Approximate mass of\n" "each EOI application", None
            )
        )
        self.litterInputLabel_0.setText(
            QtWidgets.QApplication.translate(
                "project", "kg dry matter per hectare", None
            )
        )
        self.label_88.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">External Organic Inputs - Type 2</span></p></body></html>',
                None,
            )
        )
        self.label_31.setText(
            QtWidgets.QApplication.translate(
                "project",
                "External Organic Inputs (EOIs)\n"
                "such as litter, mulch or manure\n"
                "added to the field",
                None,
            )
        )
        self.litterInterval_1.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.litterFreqLabel_1.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.litterCustom_1.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.litterCustomText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "List and/or range of years in which\n"
                "EOIs are added (e.g. 2,7-9,18):",
                None,
            )
        )
        self.litterInputText_1.setText(
            QtWidgets.QApplication.translate(
                "project", "Approximate mass of\n" "each EOI application", None
            )
        )
        self.litterInputLabel_1.setText(
            QtWidgets.QApplication.translate(
                "project", "kg dry matter per hectare", None
            )
        )
        self.label_59.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Synthetic Fertiliser Inputs - Type 1</span></p></body></html>',
                None,
            )
        )
        self.label_29.setText(
            QtWidgets.QApplication.translate(
                "project", "Synthetic fertiliser added to field", None
            )
        )
        self.fertInterval_0.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.fertFreqLabel_0.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.fertCustom_0.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.fertCustomText_0.setText(
            QtWidgets.QApplication.translate(
                "project",
                "List and/or range of years in which\n"
                "fertiliser is added (e.g. 2,7-9,18):",
                None,
            )
        )
        self.fertInputText_0.setText(
            QtWidgets.QApplication.translate(
                "project", "Mass of each fertiliser application", None
            )
        )
        self.fertInputLabel_0.setText(
            QtWidgets.QApplication.translate("project", "kg per hectare", None)
        )
        self.fertNitrogenText_0.setText(
            QtWidgets.QApplication.translate(
                "project", "Nitrogen content of added fertiliser:", None
            )
        )
        self.fertNitrogenLabel_0.setText(
            QtWidgets.QApplication.translate("project", "% Nitrogen", None)
        )
        self.label_87.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt; font-weight:600;">Synthetic Fertiliser Inputs - Type 2</span></p></body></html>',
                None,
            )
        )
        self.label_30.setText(
            QtWidgets.QApplication.translate(
                "project", "Synthetic fertiliser added to field", None
            )
        )
        self.fertInterval_1.setText(
            QtWidgets.QApplication.translate("project", "Every", None)
        )
        self.fertFreqLabel_1.setText(
            QtWidgets.QApplication.translate("project", "years", None)
        )
        self.fertCustom_1.setText(
            QtWidgets.QApplication.translate("project", "Other ", None)
        )
        self.fertCustomText_1.setText(
            QtWidgets.QApplication.translate(
                "project",
                "List and/or range of years in which\n"
                "fertiliser is added (e.g. 2,7-9,18):",
                None,
            )
        )
        self.fertInputText_1.setText(
            QtWidgets.QApplication.translate(
                "project", "Mass of each fertiliser application", None
            )
        )
        self.fertInputLabel_1.setText(
            QtWidgets.QApplication.translate("project", "kg per hectare", None)
        )
        self.fertNitrogenText_1.setText(
            QtWidgets.QApplication.translate(
                "project", "Nitrogen content of added fertiliser:", None
            )
        )
        self.fertNitrogenLabel_1.setText(
            QtWidgets.QApplication.translate("project", "% Nitrogen", None)
        )
        self.end_title_label.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:18pt; font-weight:600;">END OF NEW BASELINE SCENARIO</span></p></body></html>',
                None,
            )
        )
        self.label_223.setText(
            QtWidgets.QApplication.translate(
                "project",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">Click the &quot;Save&quot; button to save information</span></p>\n'
                '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:14pt;">or click &quot;Back&quot; to review.</span></p></body></html>',
                None,
            )
        )
        self.backButton.setText(
            QtWidgets.QApplication.translate("project", "Back", None)
        )
        self.nextButton.setText(
            QtWidgets.QApplication.translate("project", "Next", None)
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    project = QtWidgets.QDialog()
    ui = Ui_project()
    ui.setupUi(project)
    project.show()
    sys.exit(app.exec_())
