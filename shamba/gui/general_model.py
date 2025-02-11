"""Functions for running the general infor section of the model from gui"""

import os
import sys

from shamba.model.common import csv_handler
try:
    from io import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
import logging as log
import numpy as np
from PyQt5 import QtCore, QtGui

import shutil

from shamba.model import configuration
from shamba.model.climate import Climate


class GeneralModel(object):

    """Run the part of the model that can be 
    run with the info from general dialog)."""

    def __init__(self, ui):
        """
        Args:
            ui: GeneralDialog instance

        """
        self.ui = ui

        # metadata
        configuration.PROJ_NAME = str(self.ui.projectName.text())
        
        # for use in the xml file for oe descriptive data
        self.farmer_name = str(self.ui.farmerName.text())
        self.area = self.ui.area.value()
        self.field_num = self.ui.fieldNum.value()

        if self.ui.projectType.currentIndex() == 0:
            self.isOneField = False
        else:
            self.isOneField = True

        # climate and location
        self.location, configuration.N_YEARS, configuration.N_ACCT = self.get_params()
        self.climate = self.get_climate()
        
        self.save_description()

    def get_params(self):
        """Read general params from the ui."""
        lat = np.round(self.ui.latitude.value(), decimals=6)
        long = np.round(self.ui.longitude.value(), decimals=6)

        location = (lat, int)
        n_years = self.ui.mitigationYears.value()
        n_acct = self.ui.mitigationYears.value()

        return location, n_years, n_acct

    def get_climate(self):
        """Get climate data from the ui."""
        if self.ui.climFromCRU.isChecked():
            climate = Climate.from_location(self.location)
        elif self.ui.climFromCsv.isChecked():
            src_file = str(self.ui.climCsvBrowseBox.text())
            self.climateFilename = src_file
            climate = Climate.from_csv(src_file)
        elif self.ui.climFromCustom.isChecked():
            climate = Climate.from_csv(self._print_climate_from_ui())
        else:
            log.error("Climate data input method not selected")
            sys.exit(2)

        return climate

    def _print_climate_from_ui(self):
        """
        Print the climate data in the ui table to csv
        (called if 'custom climate data' was selected)
        """
        cols = self.ui.climateTable.columnCount()
        rows = self.ui.climateTable.rowCount()

        clim = []
        for c in range(cols):
            clim.append([])
            for r in range(rows):
                item = self.ui.climateTable.item(r, c)
                try:
                    item = float(item.text())
                except AttributeError:  # field is empty
                    log.warning(
                            "Empty field in climate table - was set to 0")
                    item = 0
                except ValueError:  # non-numeric field
                    log.exception(
                            "Non-numeric value in climate table - set to nan")
                    item = np.nan

                clim[c].append(item)

        clim = np.array(clim)

        if self.ui.openPanEvapCheck.isChecked():
            evap = True
        elif self.ui.evapotransCheck.isChecked():
            clim[2] /= 0.75     # convert to evap
        else:
            log.error("ONE OF EVAP/PET not selected")
            sys.exit(2)

        # Save climate to csv in INP_DIR then init climate object from that
        filepath = os.path.join(configuration.INP_DIR, 'climate.csv')
        csv_handler.print_csv(
                filepath,
                 np.transpose(clim),
                col_names=['temp', 'rain', 'evap'],

        )
        return filepath

    def save_data(self):
        """Save info. Called in shamba.pyw as a slot."""
        
        # climate
        self.climate.save_()
        
        # general info
        filename = os.path.join(configuration.OUT_DIR, 'general_info.txt')
        with open(filename, 'w+') as fout:
            fout.write(self.description)

        # climate input file, if it exists
        try:
            dest_file = os.path.basename(self.climateFilename)
            dest_file = os.path.join(configuration.INP_DIR, dest_file)
            shutil.copyfile(self.climateFilename, dest_file)
        except AttributeError:  # no climate input file
            pass

    def save_description(self):
        """print general info to description string"""
        # redirect stdout to a StringIO object
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        print ("GENERAL PROJECT INFORMATION")

        print ("\nOVERVIEW")
        print("Project name:\n\t%s" % configuration.PROJ_NAME)
        print(("Level of assessment:\n\t%s" % str(
                self.ui.projectType.currentText())))
        if self.ui.projectType.currentIndex() != 0:
            print("Name of farmer:\n\t%s" % str(self.ui.farmerName.text()))
            print("Field number:\n\t%d" % self.ui.fieldNum.value())
            print("Field area:\n\t%d ha" % (self.ui.area.value()))

        print ("\nLocation and Project Periods")
        print(("Project location:\n\t(lat,long) = (%f, %f)" % (
                self.location[0], self.location[1])))
        print("Quantification period:\n\t%d years" % configuration.N_ACCT)

        print ("\nCLIMATE")
        print ("Climate data loaded from:")
        if self.ui.climFromCRU.isChecked():
            print ("\tSHAMBA default data")
        elif self.ui.climFromCsv.isChecked():
            print("\t%s" % self.climateFilename)
        else:
            print ("\tcustom data")
        self.climate.print_()
        
        # revert back to old stdout
        sys.stdout = old_stdout
        self.description = mystdout.getvalue()
        mystdout.close()
