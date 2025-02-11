from PyQt5 import QtCore, QtGui

from shamba.gui.translate_ import translate_ as _
from shamba.gui import designer
from shamba.gui.designer.disclaimer_ui import Ui_Disclaimer
import sys
import os


DISCLAIMER = """The SHAMBA tool has been developed as part of the Climate-Smart Agriculture in Sub-Saharan Africa project, led by the Sustainability and Climate Change team at PwC UK, with funding from the Rockefeller Foundation. It was created by members of the tropical land use research group in the School of Geosciences at the University of Edinburgh. 

The outputs of the tool are designed to give an indication of the mitigation potential of implementing CSA practices at specified sites. This information can be used to help assess the feasibility of a project or to provide an indication of potential project performance. The functionality for monitoring emissions reductions and removals achieved by a project are not available in this version of the SHAMBA tool.

The SHAMBA tool should be used in conjunction with the SHAMBA tool user guide. Users should click the OK button below in acknowledgment of  the following terms of use:

This tool has been prepared for general guidance on matters of interest only, and does not constitute professional advice. You should not act upon the information contained in this tool without obtaining specific professional advice. No representation or warranty (express or implied) is given as to the accuracy or completeness of the information contained in this tool, and, to the extent permitted by law, the authors, publishers and distributors of this tool do not accept or assume any liability, responsibility or duty of care for any consequences of you or anyone else acting, or refraining to act, in reliance on the information contained in this tool or for any decision based on it.
"""

class DisclaimerDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Disclaimer()
        self.ui.setupUi(self)
     
        designer_dir = os.path.dirname(os.path.abspath(designer.__file__))
        image_dir = os.path.join(designer_dir, 'splash.jpg')
        pix = QtWidgets.QPixmap(image_dir)
        self.ui.splashImage.setPixmap(pix.scaled(self.ui.splashImage.size(),QtCore.Qt.KeepAspectRatio))
        self.ui.disclaimerText.setText(_(DISCLAIMER))

    def reject(self):
        sys.exit(0)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myapp = DisclaimerDialog()
    myapp.show()
    sys.exit(app.exec_())
