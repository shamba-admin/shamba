"""
Module containing translate_ function.
Just a shorthand for QGui.QApplication.translate_
"""

from PyQt5 import QtGui, QtWidgets


def translate_(text):
    """
    Add a QString to the list of translatable strings for the project.
    Acts as wrapper to QAppplication.translate_ function
    to save space basically
    """

    return QtWidgets.QApplication.translate("MainWindow", text, None)
