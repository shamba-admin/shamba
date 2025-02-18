#!/usr/bin/python

"""Contains class for setting up the general info dialog."""


import os
from model import configuration

from PyQt5 import QtCore, QtGui
from gui.translate_ import translate_ as _
from gui.designer.general_dialog_ui import Ui_general
from gui.page_change import NextBackButtons
from gui.dialog_setup import GenericDialog, GenericPage


class GeneralDialog(GenericDialog, Ui_general):
    """
    Set up the General dialog that pops up when
    "enter general project information" is pressed

    Set up things like toggling of items/pages, which items are 'mandatory'
    , etc. for each page in the dialog's stackedWidget

    Extends the GenericDialog and Ui_general classes.
    """

    def __init__(self, parent=None):
        super(GeneralDialog, self).__init__(parent)
        self.setupUi(self)  # bare ui from designer

        self.buttons = NextBackButtons(
            self.nextButton, self.backButton, self.stackedWidget
        )

        # Make list of all the pages in the stackedWidget
        sw = self.stackedWidget
        sw.setCurrentIndex(0)
        for w in [sw.widget(i) for i in range(sw.count())]:
            self.pageList.append(w)

        self._setup_overview_page()
        self._setup_param_page()
        self._setup_climate_page()
        self._setup_climateTable_page()

        for p in self.pageList:
            for par in self.parList:
                if p not in par:
                    par[p] = []
            GenericPage(
                p,
                sw,
                self.buttons,
                toComplete=self.toComplete[p],
                toCompleteConditional=self.toCompleteConditional[p],
                toToggle=self.toToggle[p],
                buttonsToShowPage=self.buttonsToShowPage[p],
                buttonsToHidePage=self.buttonsToHidePage[p],
                buttonGroups=self.buttonGroups[p],
                comboBoxes=self.comBox[p],
            )

        # for help text
        self.helpButton.clicked.connect(lambda: self.show_help(sw.currentWidget()))

    def _setup_overview_page(self):
        p = self.overviewPage
        self.toToggle[p] = {
            self.projectType: [
                self.farmerName,
                self.farmerNameLabel,
                self.fieldNumLabel,
                self.fieldNum,
                self.areaLabel,
                self.area,
                self.areaText,
            ]
        }

        self.help_text[p] = _(
            (
                "Enter the name of the project and select whether you "
                "will be using the tool to assess mitigation potential "
                "at the individual field or project level."
            )
        )

        self.toComplete[p] = [(self.projectName, type(self.projectName))]

    def _setup_param_page(self):
        p = self.paramPage
        self.help_text[p] = _(
            (
                "Enter the latitude and longitude of the Project "
                "Intervention Area in decimal degrees. The location "
                "can be determined from a GPS reading made in the centre "
                "of the Project Intervention Area (make sure the units "
                "of the GPS are set to decimal degrees), or from an online "
                "map such as Google Earth. To set the units in Google Earth "
                "to decimal degrees select Tools>Options... and in the 3D "
                'View tab ensure that the "Show Lat/Long" tab is set to '
                "decimal degrees. "
                "\n\nThe Quantification Period is the period over which the "
                "climate benefit of the project intervention is assessed."
            )
        )

        # self.toComplete[p] = [
        #        self.latitude, self.longitude,
        #        self.modelYears, self.mitigationYears]

    def _setup_climate_page(self):
        p = self.climatePage

        self.help_text[p] = _(
            (
                "SHAMBA default climate data is from the CRU-TS dataset. "
                "Using the default data is recommended for most users, "
                "unless high quality local climate data are available - for "
                "example from a local weather station."
                "\n\nIf you chose to load climate data, the file must be "
                "in .csv format and match the format of the sample climate "
                "file (found in shamba/sample_input_files/climate.csv."
                "Alternatively, you can choose to manually enter the data "
                "in a table on the next page."
            )
        )

        self.buttonGroups[p] = [
            [self.climFromCRU, self.climFromCsv, self.climFromCustom]
        ]
        self.toToggle[p] = {
            self.climFromCsv: [
                self.climCsvLabel,
                self.climCsvBrowseBox,
                self.climCsvBrowseButton,
            ]
        }
        self.toCompleteConditional[p] = [
            (self.climCsvBrowseBox, type(self.climCsvBrowseBox), self.climFromCsv)
        ]
        # file browser for climate
        self.climCsvBrowseButton.clicked.connect(
            lambda: self.browse_for_file(self.climCsvBrowseBox)
        )

    def _setup_climateTable_page(self):
        p = self.climateTablePage
        self.help_text[p] = _(
            (
                "Enter custom climate data in the table. "
                '\n\n"Which are you entering?" Check the corresponding box '
                "for whether you are entering Open-pan evaporation or "
                "evapotranspiration data in the third column. "
            )
        )

        self.buttonGroups[p] = [[self.openPanEvapCheck, self.evapotransCheck]]
        self.buttonsToShowPage[p] = [self.climFromCustom]
        self.buttonsToHidePage[p] = [self.climFromCRU, self.climFromCsv]

        # Stuff for setting the text in the column labels
        self.climateTable.horizontalHeaderItem(0).setText(
            _("Temperature (" + "\N{DEGREE CELSIUS}" + ")")
        )
        self.climateTable.horizontalHeaderItem(2).setText(_(""))

        # Set row label based on whether PET or evap selected
        self.openPanEvapCheck.clicked.connect(
            lambda: self.climateTable.horizontalHeaderItem(2).setText(
                _("Open-pan evaporation (mm)")
            )
        )
        self.evapotransCheck.clicked.connect(
            lambda: self.climateTable.horizontalHeaderItem(2).setText(
                _("Evapotranspiration (mm)")
            )
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = GeneralDialog()
    sys.exit(ui.exec_())
