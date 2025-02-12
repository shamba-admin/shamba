

from PyQt5 import QtCore, QtGui, QtWidgets
from gui.translate_ import translate_ as _
from gui.designer.name_dialog_ui import Ui_name


class NameDialog(Ui_name, QtWidgets.QDialog):

    def __init__(self, isBaseline=True, parent=None):
        super(NameDialog, self).__init__(parent)
        self.setupUi(self)
        if not isBaseline:
            self.name_label.setText(
                    _("Enter a name for the new intervention"))
            self.setWindowTitle(_("Intervention"))

        self.buttonBox.buttons()[0].setEnabled(False)
        self.nameBox.textChanged.connect(
                lambda: self._check_complete(self.nameBox.text())
        )
    
    @QtCore.pyqtSlot()
    def _check_complete(self, text):
       
        if len(str(text).strip()) > 0:
            self.buttonBox.buttons()[0].setEnabled(True)
        else:
            self.buttonBox.buttons()[0].setEnabled(False)
