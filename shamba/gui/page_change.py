"""Module containing NextBackButtons and PageComplete classes"""

import math
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets

from gui.translate_ import translate_ as _


class NextBackButtons(QtCore.QObject):
    """
    Class for dealing with properties and slots related to the
    next and back buttons in a QStackedWidget.
    """

    # Emitted after next button is pressed, *before* page actually changes
    finishedPage = QtCore.pyqtSignal(QtWidgets.QWidget)

    # note also that the 'accepted' signal of the parent QDialog
    # is emitted when 'Done' button is pressed
    def __init__(self, nextButton, backButton, stack, parent=None):
        """
        Initialise object.
        Arguments:
            nextButton: QPushButton that moves page forward when pressed
            backButton: QPushButton that moves page back when pressed
            stack: QStackedWidget that the buttons navigate
                    - parent widget (not neccessarily directly) of buttons

        """

        super(NextBackButtons, self).__init__(parent)

        self.nextButton = nextButton
        self.backButton = backButton
        self.stack = stack
        self.dialog = self.stack.parent()  # parent QDialog

        self.connectSlots()

    def connectSlots(self):
        """
        Connect the custom slots for page-turning
        to the button.clicked signals.
        """
        self.nextButton.clicked.connect(self.goToNextPage)
        self.backButton.clicked.connect(self.goToPrevPage)

        # Each widget in the stack must have a dict of bools named complete
        # as an attribute so that buttons can be disabled/enabled
        self.stack.currentChanged.connect(
            lambda: self.enableButtons(
                all(self.stack.currentWidget().complete.isComplete.values())
            )
        )
        self.enableButtons(True)

    def disconnectSlots(self):
        """
        Disconnect the page-turning slots
        from the button.clicked signal.
        """
        self.nextButton.clicked.disconnect()
        self.backButton.clicked.disconnect()
        self.stack.currentChanged.disconnect()

    @QtCore.pyqtSlot(bool)
    def enableButtons(self, isComplete):
        """
        Enable/disable 'next' button depending on if the page
        is complete, and disable 'back' button if on first page.
        """
        self.nextButton.setEnabled(isComplete)

        i = self.stack.currentIndex()
        if i == 0:
            self.backButton.setEnabled(False)
        else:
            self.backButton.setEnabled(True)

        # set the next button text to "Next"
        # since it could have been changed to "Save" for the last page
        # Can't be "Save" after pressing the back button though
        # since Save can only be on last page of the stack
        self.nextButton.setText(_("Next"))

    @QtCore.pyqtSlot()
    def goToNextPage(self):
        """
        Set the current page of the stackedWidget
        forward one page.

        Slot connected the nextButton.clicked signal.

        Emits 'finishedPage' signal
        """

        w = self.stack.currentWidget()
        i = self.stack.currentIndex()

        if i < self.stack.count() - 1:
            self.finishedPage.emit(w)
            self.stack.setCurrentIndex(i + 1)

        else:
            # QMessageDialog to confirm if user wants to finish
            # accept the dialog
            self.dialog.accept()

        # Set the text to "Save" if going on to last page
        if i == self.stack.count() - 2:
            self.nextButton.setText(_("Save"))
        else:
            self.nextButton.setText(_("Next"))

    @QtCore.pyqtSlot()
    def goToPrevPage(self):
        """
        Set current page in stackedWidget back one page

        Slot connected to backButton.clicked signal.
        """

        w = self.stack.currentWidget()
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        else:
            pass


class PageComplete(QtWidgets.QWidget):
    """

    Class for checking if page of stackedLayout has been completed
    (when the values of given list of objects
     on the page have been completed)
    i.e. 'required fields'

    """

    def __init__(self, stackedWidget, buttons, toComplete=[], toCompleteConditional=[]):
        """
        Initialise object.
        Args:
            toComplete: list with the QWidgets to complete
                        format of each list item is (widget, type(widget))
            toCompleteConditional: same as toComplete, but 3rd item
                                   in tuple is the button which toggles
                                   whether this widget needs completing
            pageWidget: page in question
            stackedWidget: currentStackedWidget
            buttons: NextBackButtons object so buttons can be enabled
                     when page is completed


        """
        super(PageComplete, self).__init__()
        self.buttons = buttons
        self.sw = stackedWidget
        self.toComplete = toComplete
        self.toCompleteConditional = toCompleteConditional

        # Store completeness bools for each widget
        self.isComplete = {}
        for widget, type in toComplete:
            self.isComplete[widget] = False
        for widget, type, button in toCompleteConditional:
            self.isComplete[widget] = True

        # connect signals and slots for toComplete
        self.connect_widgets()
        for widget, type, button in self.toCompleteConditional:
            button.toggled.connect(self._toggle_toComplete)

        # Enable buttons - start on title page
        self.buttons.enableButtons(True)

    def connect_widgets(self, connectBool=True):
        """
        Create dict with the signal and slot combo for each item
        on the toComplete list.
        e.g. a QSpinBox will have signal=valueChanged,
                slot=_check_complete_spinbox

        Connects the signals/slots so that when the item is changed
        (e.g. valueChanged, buttonClicked, etc.), the item can be marked
        as 'complete'.

        connectBool=True if to be connect, False if to be disconnected"""
        # Store what signal and slot to use for particular type

        def signal(widget, type):
            # Returns the signal name (callable) and the arguments to emit
            if type == QtWidgets.QLineEdit:
                sig = widget.textChanged
                emit = widget.text()
            elif type == QtWidgets.QSpinBox or type == QtWidgets.QDoubleSpinBox:
                sig = widget.valueChanged
                emit = widget.value()
            elif type == QtWidgets.QButtonGroup:
                sig = widget.buttonClicked
                emit = widget.checkedButton()
            elif type == QtWidgets.QComboBox:
                sig = widget.currentIndexChanged
                emit = widget.currentIndex()
            return sig, emit

        slot = {
            QtWidgets.QLineEdit: self._check_complete_QLineEdit,
            QtWidgets.QSpinBox: self._check_complete_QSpinBox,
            QtWidgets.QDoubleSpinBox: self._check_complete_QDoubleSpinBox,
            QtWidgets.QButtonGroup: self._check_complete_QButtonGroup,
            QtWidgets.QComboBox: self._check_complete_QComboBox,
        }

        # Connect signals and slots for checking completeness
        for widget, type in self.toComplete:
            sig = signal(widget, type)
            if connectBool:
                sig[0].connect(slot[type])
                sig[0].emit(sig[1])
            else:
                sig[0].disconnect()

    def _toggle_toComplete(self, checked):
        """
        Go through the toCompleteConditional list and update
        whether or not they need to be completed
        (based on if the conditional button is toggled)

        """
        self.connect_widgets(False)

        # iterate over tuples that have sender() in them
        for widget, type, button in [
            x for x in self.toCompleteConditional if self.sender() in x
        ]:
            if checked:
                self.toComplete.append((widget, type))
            else:
                try:
                    self.toComplete.remove((widget, type))
                    self.isComplete[widget] = True
                except ValueError:  # not in list
                    pass
        self.connect_widgets(True)

    # Slots to check completeness of various types of objects
    def _check_complete_QLineEdit(self, text):
        """Check completeness of a QLineEdit -
        complete when changed to a non-blank value
        """
        # Check if lineEdit itself is actually done
        if len(str(text).strip()) == 0:
            # line is blank (strip accounts for lines with just spaces)
            self.isComplete[self.sender()] = False
            self.buttons.enableButtons(False)
        else:
            self.isComplete[self.sender()] = True
            self.buttons.enableButtons(all(self.isComplete.values()))

    def _check_complete_QDoubleSpinBox(self, value):
        """Check completeness of a QDoubleSpinBox -
        complete when changed to non-zero value
        """
        if math.fabs(value) < 0.00000001:
            # sufficiently close to 0
            self.isComplete[self.sender()] = False
            self.buttons.enableButtons(False)
        else:
            self.isComplete[self.sender()] = True
            self.buttons.enableButtons(all(self.isComplete.values()))

    def _check_complete_QSpinBox(self, value):
        """Check completeness of a QSpinBox -
        complete when changed to non-zero value
        """
        if value == 0:
            self.isComplete[self.sender()] = False
            self.buttons.enableButtons(False)
        else:
            self.isComplete[self.sender()] = True
            self.buttons.enableButtons(all(self.isComplete.values()))

    def _check_complete_QButtonGroup(self, buttonPressed):
        """Check completeness of a QButtonGroup -
        complete when this slot is called
        """
        if buttonPressed is None:  # no button checked when buttonClicked
            return  # emitted

        self.isComplete[self.sender()] = True
        self.buttons.enableButtons(all(self.isComplete.values()))

    def _check_complete_QComboBox(self, index):
        """Check completeness of a QComboBox -
        complete when anything but index 0 is selected
        (make sure index 0 is always a title)
        """
        if index == 0:
            self.isComplete[self.sender()] = False
            self.buttons.enableButtons(False)
        else:
            self.isComplete[self.sender()] = True
            self.buttons.enableButtons(all(self.isComplete.values()))
