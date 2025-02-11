#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import shutil
from distutils.dir_util import copy_tree
from xml.etree import ElementTree as ET
from xml.dom import minidom

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)
sys.path.append(os.path.dirname(_dir))


from shamba.model import configuration, io, io_handler

from PyQt5 import QtGui, QtCore, QtWidgets

from shamba.gui.designer.main_ui import Ui_SHAMBA
from shamba.gui.general_dialog import GeneralDialog
from shamba.gui.project_dialog import ProjectDialog
from shamba.gui.name_dialog import NameDialog
from shamba.gui.translate_ import translate_ as _
from shamba.gui.page_change import NextBackButtons

from shamba.gui.general_model import GeneralModel
from shamba.gui.project_model import ProjectModel
from shamba.gui.emit_model import Emissions


class MainInterface(QtWidgets.QMainWindow, Ui_SHAMBA):
    
    """
    Main driver for the SHAMBA tool (GUI mode).
    
    Sets up the ui from designer files and connects signals to show the 
    general and project dialogs, as well as the mitigation graphs

    Inherits QMainWindow and extends the Ui_SHAMBA class (from designer).

    """

    def __init__(self, parent=None):
        super(MainInterface, self).__init__(parent) # init QMainWindow
        
        self.setupUi(self)  # init Ui from designer 
        self.show()       
        
        # initial setup
        self.model = Model(self)
        self.actionSave.triggered.connect(self.model.save_)
        self.actionSave_As.triggered.connect(self.model.save_as)
        
        self._setup_error_messages()
        
        # Set up help stuff
        self.help_text = {}
        self.help_text[self.mitigationTab] = _((
                "Specify the baseline scenario and project intervention "
                "you want to compare. The graph shows the annual greenhouse "
                "gas emissions per hectare per year over the specified "
                "accounting period. The summary next to the graph shows "
                "total emissions or removals of greenhouse gases over the "
                "accounting period (a negative value indicates a removal "
                "of greenhouse gases from the atmosphere). The Net impact "
                "is the difference between the emissions/removals in the "
                "baseline and intervention scenarios. This is the mitigation "
                "potential of the intervention."
                "\n\nClick 'Save graph' to save an image file (png format) "
                "of the graph that is produced. The graph will not be saved "
                "automatically when the project is saved (file->save) "
                "and must be saved manually."
        ))
        self.helpButton.clicked.connect(
                lambda: self.show_help(self.tabWidget.currentWidget())
        )

        # Signals to show general info dialog
        self.generalButton.clicked.connect(self.show_general_dialog)
        
        # Signals to show the project dialog
        self.baselineButton.clicked.connect(
                lambda: self.show_project_dialog(
                                self.tabWidget.currentWidget(), 
                                self.model.baselines,
                                isBaseline=True)
        )
        self.interventionButton.clicked.connect(
                lambda: self.show_project_dialog(
                                self.tabWidget.currentWidget(), 
                                self.model.interventions,
                                isBaseline=False)
        )
        
        # Signal to move between pages in the two stackedWidgets
        bs = self.baselineSelector
        bs.currentIndexChanged.connect(
                lambda: self.baselineOutputBox.setText(
                        self.model.baselines[bs.currentIndex()].description)
        )
        is_ = self.interventionSelector
        is_.currentIndexChanged.connect(
                lambda: self.interventionOutputBox.setText(
                        self.model.interventions[is_.currentIndex()].description)
        )
        
        # sister comboBoxes that change with their siblings
        self.sisterComboBox = {
                self.baselineSelector: self.baselineSelectorPlot,
                self.interventionSelector: self.interventionSelectorPlot
        }

        # Signal for plotting mitigation estimates
        self.mitigationButton.clicked.connect(self.show_mitigation_graph)
        self.graphSaveButton.clicked.connect(self.model.save_graph)

   # #@QtCore.pyqtSlot() https://stackoverflow.com/questions/40674940/why-does-pyqtslot-decorator-cause-typeerror-connect-failed
    def show_help(self, tab):
        """Open a QMessageBox.about dialog to show help text

        Args:
            tab: the currnent tab (in the stackedWidget) to show help for

        """
        try:
            text = self.help_text[tab]
        except KeyError:
            text = "No help available."

        help_dialog = QtWidgets.QMessageBox(tab.parent())
        help_dialog.setModal(False)
        help_dialog.setWindowTitle(_("Help"))
        help_dialog.setText(_(text))
        help_dialog.show()
 

    ##@QtCore.pyqtSlot()
    def show_general_dialog(self):
        """
        Slot to show the general dialog (in general_dialog.GeneralDialog)
        
        Also  call GeneralModel (general_model.GeneralModel) 
        when dialog exits properly.
        """
        # Show "are you sure" dialog if general info already entered"
        if self.model.general is not None:
            popup = QtWidgets.QMessageBox(self)
            popup.setWindowTitle(_("Overwrite?"))
            popup.setText(_(
                    "This will overwrite previous " + 
                    "general project information"))
            popup.setInformativeText(_(
                   "Do you wish to continue?"))
            popup.setStandardButtons(
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if popup.exec_() == QtWidgets.QMessageBox.No:
                return
        
        # clear any previous data
        self.generalOutputBox.clear()

        # show general info
        generalUi = GeneralDialog(parent=self)
        if generalUi.exec_():   # ended properly (by pressing 'Done')
      
            #sys.stdout = OutLog(self.generalOutputBox, sys.stdout)
            _show_busy_cursor()
            
            try:
                self.model.general = GeneralModel(generalUi)
                self.model._update_save_status(False)
            except io.FileOpenError:
                self.climateOpenError.showMessage(_(
                        "Could not open climate file. " + 
                        "Please try completing this form again."))
            
            self.generalOutputBox.setText(self.model.general.description)
            _show_normal_cursor()
    
    #@QtCore.pyqtSlot(QtWidgets.QWidget, list)
    def show_project_dialog(self, tab, model_list, isBaseline=True):
        """
        Slot to show the project dialog (project_dialog.ProjectDialog)

        Also calls model (project_model.ProjectModel)
        when dialog exits properly.
        
        Args:
            stack: widget which has a QComboBox and QStackedWidget as 
                   children, so that new project can be added to these.
            model_list: list to append this project to 
                        (e.g. model.baselines or model.interventions)

        """ 
        # Show error popup window if a general dialog has
        # not been completed first
        if self.model.general is None:
            self.noGeneralInfoError.showMessage(_(
                    "You must enter general project information before " +
                    "a baseline or intervention can be added")
            )
            return
        
        nameDialog = NameDialog(isBaseline=isBaseline, parent=self)
        if not nameDialog.exec_():   # cancelled
            return
        else: 
            projectName = nameDialog.nameBox.text()
        
        # Find the right comboBox in the tab (baseline or intervention)
        selector = tab.findChild(QtWidgets.QComboBox)
        
        # run the project dialog
        projectUi = ProjectDialog(isBaseline=isBaseline, parent=self)
        if projectUi.exec_():   # dialog accepted

            # application is busy
            _show_busy_cursor()
            
            # find index and the outputBox to write info in
            index = selector.currentIndex()
            outputBox = tab.findChild(QtWidgets.QTextEdit)
            
            # Actually run the model
            model_list.append(
                ProjectModel(
                        projectUi, self.model.general, 
                        projectName, index+1, isBaseline)
            )
            self.model._update_save_status(False)
           
            # set the output text and add the project to the combobox
            outputBox.setText(model_list[-1].description)
            self._add_project_in_gui(projectName, selector)

            # Return to normal
            _show_normal_cursor()
            
    #@QtCore.pyqtSlot()
    def show_mitigation_graph(self):
        """
        Slot to show the mitigation estimates plot
        for a baseline and intervention.
        
        First calculates mitigation estimates by calling
        the emissions model (emit_mode.Emissions)

        """

        
        # Show error if no baseline or intervention created yet
        curr_text_b = str(self.baselineSelectorPlot.currentText())
        curr_text_i = str(self.interventionSelectorPlot.currentText())
        if (
            (curr_text_b.startswith("<") and curr_text_b.endswith(">")) or 
            (curr_text_i.startswith("<") and curr_text_i.endswith(">"))
        ):
            self.missingData.showMessage(_(
                    "Baseline or intervention missing"))
            return
        else:
            _show_busy_cursor()

            # select the right baseline and intervention 
            baseline = self.model.baselines[
                    self.baselineSelectorPlot.currentIndex()]
            intervention = self.model.interventions[
                    self.interventionSelectorPlot.currentIndex()]
            base_name = baseline.name
            inter_name = intervention.name

            # Run emission model (also includes soil model)
            # for baseline and intervention
            # If that's been done for this set of baseline-intervention,
            # load the saved one
            for em in self.model.emissions:
                if (baseline.name == em.baseline.name and 
                        intervention.name == em.intervention.name):
                    current_emissions = em
                    break
            else:
                current_emissions = Emissions(
                        self, self.model.general, baseline, intervention)
                self.model.emissions.append(current_emissions)
                self.model._update_save_status(False)

            # Remove baseline and project when new one is plotted
            # removePlot method handles if there are no plots yet
            self.plotWidget.removePlot(0)
            self.plotWidget.removePlot(1)


            self.plotWidget.addPlot(
                    current_emissions.emit_base.emissions,
                    0,
                    baseline.name
            )
            self.plotWidget.addPlot(
                    current_emissions.emit_inter.emissions,
                    1,  
                    intervention.name
            )
            _show_normal_cursor()

    def _add_project_in_gui(self, name, selector):
        """Add a new baseline or intervention to the appropriate
        QComboBox in the gui.

        Args: 
            name: QString with project name.
            selec: QComboBox which signals the page change
            
        """
        curr_text = str(selector.currentText())
        if curr_text.startswith("<") and curr_text.endswith(">"):
            # First project
            # potential bug here if the user entered project is of form
            # "<project name>"
            for sel in [selector, self.sisterComboBox[selector]]:
                sel.setItemText(0, name)
                sel.setCurrentIndex(0)
        else:
            for sel in [selector, self.sisterComboBox[selector]]:
                sel.insertItem(selector.count(), name) # append
                sel.setCurrentIndex(selector.count()-1)

    def mousePressEvent(self, event):
        """ Override mousePressEvent so field is cleared when a 
        non-widget item in the window is clicked."""
        focusedWidget = QtWidgets.QApplication.focusWidget()
        if focusedWidget is not None:
            focusedWidget.clearFocus()
        QtWidgets.QMainWindow.mousePressEvent(self, event)

    def closeEvent(self, event):
        """
        Override closeEvent to show a dialog prompting user to 
        save, cancel, or exit without saving before quitting.
        """
        
        if self.model.save_up_to_date:
            event.accept()
        else:
            ret = self.show_save_dialog()
            if ret == QtWidgets.QMessageBox.Save:
                wasSaved = self.model.save_()
                if wasSaved:
                    event.accept()
                else:
                    event.ignore()
            elif ret == QtWidgets.QMessageBox.Discard:
                event.accept()
            else:   # cancelled
                event.ignore()
       
    def show_save_dialog(self):
        """Popup dialog that shows on a close event."""
        popup = QtWidgets.QMessageBox(self)
        popup.setWindowTitle(_("Save changes?"))
        popup.setText(_(
                "Do you want to save your changes before exiting?"))
        popup.setStandardButtons(
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel)
        ret = popup.exec_() 
        return ret

    def _setup_error_messages(self):
        """Set up the various QErrorMessage objects used."""
        self.noGeneralInfoError = QtWidgets.QErrorMessage(self)
        self.noGeneralInfoError.setWindowTitle(
                _("No general information found")
        )
        
        self.missingData = QtWidgets.QErrorMessage(self)
        self.missingData.setWindowTitle(
                _("Missing data"))
        
        self.climateOpenError = QtWidgets.QErrorMessage(self)
        self.climateOpenError.setWindowTitle(
                _("Error opening file"))
        
# Global functions for different cursor settings
def _show_busy_cursor(): 
    QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.WaitCursor))
    

def _show_normal_cursor():
    QtWidgets.QApplication.restoreOverrideCursor()


class Model(object):
    
    """Object to store model info for a gui run."""
    def __init__(self, ui):
        """
        init the Model object.

        Args:
            ui: MainInterface instance

        """
        self.ui = ui
       
        # To be populated when dialog forms are filled out in the gui
        self.general = None
        self.baselines = []
        self.interventions = []
        self.emissions = []

        # to determine if model is saved already
        self.save_filename = None
        self.save_up_to_date = True

    def get_save_filename(self):
        """Open a getSaveFileName and save the filename returned."""
        
        filename = QtWidgets.QFileDialog.getSaveFileName(
                self.ui, _('Save file'),
                os.path.abspath(configuration.PROJ_DIR),
                _("SHAMBA project files (*.shamba)")
        )
            
        if filename:
            self.save_filename =  filename + ".shamba"
            wasSaved = True  # saved
        else: # Cancelled
            wasSaved = False # not saved

        return wasSaved
    
    #@QtCore.pyqtSlot()
    def save_(self):
        """Slot to save the model information from this model run.
        Saves to existing save name if it exits"""
        if configuration.PROJ_NAME is None:
            # no general info yet
            self.ui.noGeneralInfoError.showMessage(_(
                    "Nothing to save."))
            return
        
        # create new folder if it doesn't exist yet
        if self.save_filename is None:
            wasSaved = self._create_save_dirs()
            if not wasSaved:
                return False

        # run save functions
        self._save_files()

        # cleanup
        self._update_save_status(True)
        return True
    
    #@QtCore.pyqtSlot()
    def save_as(self):
        """Like save_ method, but always save to a new name."""
        if configuration.PROJ_NAME is None:
            # no general info yet
            self.ui.noGeneralInfoError.showMessage(_(
                    "Nothing to save."))
            return
        
        # create new save folder
        wasSaved = self._create_save_dirs()
        if not wasSaved:
            _show_normal_cursor()
            return False
        
        # save files
        self._save_files()

        # cleanup
        self._update_save_status(True)
        return True

    def _create_save_dirs(self):
        """Create new save dirs from name given by user."""
        # create the save folder name if not defined yet
        # let user choose where to save
        filename_ret = self.get_save_filename()
        if not filename_ret:
            return False # not saved
        
        configuration.SAV_DIR = str(self.save_filename).split(".shamba")[0]
        try:       
            os.makedirs(configuration.SAV_DIR)
        except OSError:
            popup = QtWidgets.QMessageBox(self.ui)
            popup.setWindowTitle(_("Overwrite?"))
            popup.setText(
                    _("That folder exists. Do you want to overwrite it?"))
            popup.setStandardButtons(
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            popup_ret= popup.exec_()
            if popup_ret == QtWidgets.QMessageBox.No:
                return self._create_save_dirs() # get new filename
            else: # remove existing dir then make new one
                shutil.rmtree(configuration.SAV_DIR)
                os.makedirs(configuration.SAV_DIR)
        
        # make the rest of the dirs
        configuration.INP_DIR = os.path.join(configuration.SAV_DIR, 'input')
        configuration.OUT_DIR = os.path.join(configuration.SAV_DIR, 'output')
        os.makedirs(configuration.INP_DIR)
        os.makedirs(os.path.join(configuration.OUT_DIR, 'baselines'))
        os.makedirs(os.path.join(configuration.OUT_DIR, 'interventions'))
        return True

    def _save_files(self):
        """Actually save the different models, etc. to files."""
        self.general.save_data()
        for b in self.baselines:
            b.save_data()
        for i in self.interventions:
            i.save_data()
        for e in self.emissions:
            e.save_data()
        
        # save metadata and oe data
        io_handler.print_metadata()
        self.save_oe_info()
        for b in self.baselines:
            for i in self.interventions:
                if b.total_emissions and i.total_emissions:
                    self.save_oe_data(b,i)

    def save_oe_info(self):
        """Save the data from the tool to xml for OE."""
        project = ET.Element("project")
        name = ET.SubElement(project, "name")
        name.text = configuration.PROJ_NAME
        if self.general.isOneField:
            farmer = ET.SubElement(project, "farmer")
            farmer.text = self.general.farmer_name
            field_num = ET.SubElement(project, "field number")
            field_num.text = str(self.general.field_num)
            area = ET.SubElement(project, "field area")
            area.text = str(self.general.area) + " ha"
        else:
            region = ET.SubElement(project, "region")
            region.text = self.general.farmer_name
        acct = ET.SubElement(project, "accounting period")
        acct.text = str(configuration.N_ACCT) + " years"

        # baseline and intervention info
        for b in self.baselines:
            base = ET.SubElement(project, "baseline")
            base_name = ET.SubElement(base, "name")
            base_name.text = b.name
            if b.total_emissions is not None:
                base_emit = ET.SubElement(base, "emissions")
                base_emit.text = "%.1f tCO2/ha" % (b.total_emissions)
        for i in self.interventions:
            inter = ET.SubElement(project, "intervention")
            inter_name = ET.SubElement(inter, "name")
            inter_name.text = i.name
            if i.total_emissions is not None:
                inter_emit = ET.SubElement(inter, "emissions")
                inter_emit.text = "%.1f tCO2/ha" % (i.total_emissions)
        
        self._prettify_elem(project)
        ET.ElementTree(project).write(
                os.path.join(configuration.OUT_DIR, 'oe_info.xml'), 
                encoding='utf-8')
    
    def save_oe_data(self, baseline, intervention):
        """Save pairwise data and graph."""
        base_name = baseline.name
        inter_name = intervention.name
        
        # print data for oe
        oe_fname = os.path.join(
                configuration.OUT_DIR, "oe_data_"+base_name+"_"+inter_name+".csv")
        oe_data = "lat,long,%s,%s,net\n" % (base_name, inter_name)
        oe_data += "%.6f,%.6f,%.1f,%.1f,%.1f\n" % (
                self.general.location[0], self.general.location[1],
                baseline.total_emissions, intervention.total_emissions,
                intervention.total_emissions-baseline.total_emissions)
        with open(oe_fname, 'w+') as fout:
            fout.write(oe_data)
    
    #@QtCore.pyqtSlot()
    def save_graph(self):
        # save the plot
        base_name = str(self.ui.baselineSelectorPlot.currentText())
        inter_name = str(self.ui.interventionSelectorPlot.currentText())

        graph_fname = os.path.join(
                    configuration.OUT_DIR, "emissions_"+base_name+"_"+inter_name+".png")
        self.ui.plotWidget.saveFigure(graph_fname) 
        
        _show_normal_cursor()

    def _prettify_elem(self, elem, level=0):
        """From http://norwied.wordpress.com/2013/08/27/307/ """
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._prettify_elem(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _update_save_status(self, upToDate):
        """
        Update the save status. Used in the mainWindow closeEvent 
        to determine whether to prompt user to save before quitting.
        Also used to display an asterisk in the window title if 
        info no up to date
        
        Args:
            upToDate: whether new save status is up-to-date
        
        """
        # update status
        self.save_up_to_date = upToDate
        
        # add an asterisk to window title if change made
        title = self.ui.windowTitle()
        import unicodedata
        asciiTitle = unicodedata.normalize('NFKD', title)#.encode('ascii','ignore')
        suffix = "*"

        if not upToDate:
            if not asciiTitle.endswith(suffix):    
                asciiTitle += suffix
        else:
            if asciiTitle.endswith(suffix):
                asciiTitle.remove(QtCore.QChar(suffix))
        
        self.ui.setWindowTitle(asciiTitle)

class OutLog(object):
    """Redirect an output (stdout, stderr, etc.) to a QTextEdit.
    From http://www.riverbankcomputing.com/pipermail/pyqt/2009-February/022025.html

    """

    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtWidgets.QTextCursor.End)
        self.edit.insertPlainText( m )

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)


if __name__ == '__main__':
    import sys
    from shamba.gui.disclaimer import DisclaimerDialog

    # Set up gui
    app = QtWidgets.QApplication(sys.argv)
    
    # show the disclaimer dialog
    form = DisclaimerDialog()
    form.exec_()

    # For translation
    #translator = QtCore.QTranslator()
    #translator.load("gui/designer/name_dialog_fr.qm")
    #app.installTranslator(translator)

    ui = MainInterface()
    
    sys.exit(app.exec_())

