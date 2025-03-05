"""Contains classes for all the setup of the page widgets."""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from gui.translate_ import translate_ as _
from model import configuration

from .page_change import NextBackButtons, PageComplete


class GenericDialog(QtWidgets.QDialog):
    """
    Parent class for the dialogs in the SHAMBA tool.

    Inits the items related to the page information,
    and contains methods applicable to all the dialogs.

    """

    def __init__(self, parent=None):
        super(GenericDialog, self).__init__(parent)

        # help text
        self.help_text = {}

        # To store page info
        self.pageList = []
        self.toComplete = {}
        self.toCompleteConditional = {}
        self.toToggle = {}
        self.buttonsToShowPage = {}
        self.buttonsToHidePage = {}
        self.spinboxesToShowPage = {}
        self.buttonGroups = {}
        self.comBox = {}

        self.parList = [
            self.toComplete,
            self.toCompleteConditional,
            self.toToggle,
            self.buttonsToShowPage,
            self.buttonsToHidePage,
            self.spinboxesToShowPage,
            self.buttonGroups,
            self.comBox,
        ]

    def mousePressEvent(self, event):
        """Override mousePressEvent so field is cleared when a
        non-widget item in the window is clicked."""
        focusedWidget = QtWidgets.QApplication.focusWidget()
        if focusedWidget is not None:
            focusedWidget.clearFocus()
        QtWidgets.QDialog.mousePressEvent(self, event)

    def browse_for_file(self, lineEdit):
        """Open a QFileDialog and show the chosen path in lineEdit."""
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            _("Open File"),
            os.path.abspath(configuration.PROJECT_DIR),
            _("CSV files (*.csv)"),
        )
        if filename:  # not cancelled
            lineEdit.setText(filename[0])

    def browse_for_dir(self, lineEdit):
        """Open a QFileDialog.getExistingDirectory and show chosen dir in
        lineEdit.
        """
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            _("Select Folder"),
            os.path.abspath(configuration.PROJECT_DIR),
        )
        if dirname:  # not cancelled
            lineEdit.setText(dirname)

    def show_help(self, page):
        """Open a QMessageBox.about dialog to show help text

        Args:
            page: page (in the stackedWidget) to show help for

        """
        try:
            text = self.help_text[page]
        except KeyError:
            text = "No help available."

        help_dialog = QtWidgets.QMessageBox(page.parent())
        help_dialog.setModal(False)
        help_dialog.setWindowTitle(_("Help"))
        help_dialog.setText(_(text))
        help_dialog.show()

    def sort_pages(self, sw, pageList):
        """
        Order the pages (widgets) currently in the stacked widget (sw)
         according to the order of pageList

        Done so that page order doesn't get screwed up when pages
        are removed/added to the stackedWidget

        """
        current_widgets = [sw.widget(i) for i in range(sw.count())]

        for proper_index, widget in enumerate(pageList):
            if widget in current_widgets:
                sw.insertWidget(proper_index, widget)

    def closeEvent(self, event):
        """
        Override closeEvent to show a dialog prompting the user
        "are you sure?" since all info in a dialog is lost
        after closing.
        """
        ret = self.show_sure_dialog()
        if ret == QtWidgets.QMessageBox.Yes:  # user sure about closing
            event.accept()
        else:
            event.ignore()

    def show_sure_dialog(self):
        """Popup dialog that shows on a close event."""

        popup = QtWidgets.QMessageBox(self)
        popup.setWindowTitle(_("Exit the dialog?"))
        popup.setText(
            _(
                "Are you sure you wish to exit this form? "
                "\nAll information entered will be lost."
            )
        )
        popup.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        ret = popup.exec_()
        return ret


class GenericPage(QtWidgets.QWidget):
    """
    Object to handle all the things pertaining
    to a particular page in the stackedWidget
    that can't be done (without a lot of effort)
    in qt-designer.

    """

    def __init__(
        self,
        pageWidget,
        stackedWidget,
        buttons,
        toComplete=[],
        toCompleteConditional=[],
        toToggle={},
        buttonsToShowPage=[],
        buttonsToHidePage=[],
        spinboxesToShowPage=[],
        buttonGroups=[],
        comboBoxes={},
    ):
        """
        Initialise object.

        pageWidget: widget of page in question
        stackedWidget: stacked widget where pageWidget is located
        toComplete: list of QObjects on page that need are required
                    (need to be completed before moving on to next page)
        toToggle: dict with list of objects to initially hide in the value,
                  and the button that toggles visibility as the key
        togglePageButtons: duple with buttons that toggle the visibility
                           of the page - button that shows page is first
        buttonsToShowPage: list of buttons which show the page
        buttonsToHidePage: list of buttons which hide the page
        spinboxesToShowPage: list of duples
                           with form (spinbox, minNumber)
                           which show/hide the page if spinbox changes
                           to >=Number/<minNumber respectively
        buttonGroups: a list with lists of buttons to put in a buttonGroup
        comboBoxes: dict with comboboxes as keys, and list of indexes
                    in that combobox to make unselectable
        """

        super(GenericPage, self).__init__()

        self.pageIndex = stackedWidget.indexOf(pageWidget)

        if buttonsToShowPage or buttonsToHidePage:
            self.setup_button_showpage(
                pageWidget, stackedWidget, buttons, buttonsToShowPage, buttonsToHidePage
            )
        if spinboxesToShowPage:
            self.setup_spinbox_showpage(
                pageWidget, stackedWidget, buttons, spinboxesToShowPage
            )
        if toToggle:
            self.setup_toggle(toToggle)
        if buttonGroups:
            bg = self.setup_button_groups(buttonGroups)
            for group in bg:
                # Add new groups to toComplete
                toComplete.append((group, type(group)))
        if comboBoxes:
            self.setup_combo_boxes(comboBoxes)

        # PageComplete object is an attribute of each widget.
        pageWidget.complete = PageComplete(
            stackedWidget, buttons, toComplete, toCompleteConditional
        )

    def setup_button_showpage(
        self, pageWidget, stackedWidget, buttons, buttonsToShowPage, buttonsToHidePage
    ):
        """
        Connects slots related to showing/removing a
        page in the stackedWidget when a button is toggled.
        """
        # button toggling to show page
        for s in buttonsToShowPage:
            s.toggled.connect(
                lambda: stackedWidget.insertWidget(self.pageIndex, pageWidget)
            )
        for h in buttonsToHidePage:
            h.toggled.connect(lambda: stackedWidget.removeWidget(pageWidget))

        # set initial state (if any hide buttons are initially pressed)
        if any([h.isChecked() for h in buttonsToHidePage]):
            stackedWidget.removeWidget(pageWidget)

    def setup_spinbox_showpage(
        self, pageWidget, stackedWidget, buttons, spinboxesToShowPage
    ):
        """
        Connects slots related to showing/removing
        a page in the stackedWidget when a spinBox value is changed

        page shown when value!=0, removed when value==0
        """

        def toggle_page(value, minShowNum):
            if value >= minShowNum:
                stackedWidget.insertWidget(self.pageIndex, pageWidget)
            else:
                stackedWidget.removeWidget(pageWidget)

        # connect the slots to the valueChanged signals
        # and set initial state
        for spinbox, minShowNum in spinboxesToShowPage:
            toggle_page(spinbox.value(), minShowNum)
            spinbox.valueChanged.connect(
                lambda: toggle_page(spinbox.value(), minShowNum)
            )

    def setup_toggle(self, toToggle, disable=False):
        """
        Connect slots for when buttons
        toggle other items' visibility.
        """

        # Slot for when combobox does the toggling
        def show_if_last_item(index, count, object_list):
            for object_ in object_list:
                if disable:
                    object_.setEnabled(index == count - 1)
                else:
                    object_.setVisible(index == count - 1)

        # slot for when spinbox does the toggling
        def show_if_nonzero(value, object_list):
            for object_ in object_list:
                object_.setVisible(value)

        # set up visibility toggling
        for toggler in toToggle:
            for object_ in toToggle[toggler]:
                # initially all hidden
                if disable:
                    object_.setEnabled(False)
                else:
                    object_.hide()

            if isinstance(toggler, QtWidgets.QRadioButton):
                for object_ in toToggle[toggler]:
                    if disable:
                        toggler.toggled.connect(object_.setEnabled)
                    else:
                        toggler.toggled.connect(object_.setVisible)
            elif isinstance(toggler, QtWidgets.QComboBox):
                # always last option in combobox that does the toggling
                toggler.currentIndexChanged.connect(
                    lambda: show_if_last_item(
                        toggler.currentIndex(), toggler.count(), toToggle[toggler]
                    )
                )
            elif isinstance(toggler, QtWidgets.QSpinBox):
                toggler.valueChanged.connect(
                    lambda: show_if_nonzero(toggler.value(), toToggle[toggler])
                )

    def setup_button_groups(self, buttonGroups):
        """
        Make button groups (of QRadioButtons) so that
        only one option can be selected per group.
        """

        bg = []  # to hold names of button groups
        for group in buttonGroups:
            bg.append(QtWidgets.QButtonGroup())
            for button in group:
                bg[-1].addButton(button)

        return bg

    def setup_combo_boxes(self, comboBoxes):
        """Setup the combo boxes so that some options are 'disabled'
        in the QComboBox

        Weird hack found at http://stackoverflow.com/questions/11099975/pyqt-set-enabled-property-of-a-row-of-qcombobox
        """

        for box in comboBoxes:
            for i in comboBoxes[box]:
                j = box.model().index(i, 0)
                box.model().setData(j, QtCore.QVariant(0), QtCore.Qt.UserRole - 1)
