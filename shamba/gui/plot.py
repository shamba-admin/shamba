import matplotlib.image
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PyQt5 import QtCore, QtGui, QtWidgets

from model import configuration

COLOURS = "brkwyc"
SUB_2 = "\u2082"


class PlotWidget(FigureCanvas):
    """a QtWidget and FigureCanvasAgg that displays the model plots"""

    def __init__(
        self, parent=None, name=None, width=5, height=4, dpi=100, bgcolor=None
    ):
        self.fig = Figure(
            figsize=(width, height), dpi=dpi, facecolor=bgcolor, edgecolor=bgcolor
        )
        self.axes = self.fig.add_axes(
            [0.125, 0.1, 0.6, 0.8]
        )  # self.fig.add_subplot(111)

        self.have_range = []

        self.axes.set_xlabel("Time (years)")
        self.axes.set_ylabel(
            "Greenhouse Gas \nEmissions/Removals (t CO" + SUB_2 + "e / ha)",
            multialignment="center",
        )

        self.axes.axhline(y=0.0, ls=":", color="k")
        self.axes.text(0, 0, "Emission", va="bottom")
        self.axes.text(0, 0, "Removal", va="top")
        self.emission_patch = None
        self.removal_patch = None
        self.updateShading()

        self.totalsBox = self.fig.text(0.75, 0.1, "")

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.updateGeometry()

        self.plots = {}
        self.totals = {}

    def addPlot(self, data, plotID, name):
        # No support for ranges (uncertainties) yet
        plots = []
        plots += self.axes.plot(
            np.array(list(range(len(data)))) + 1,  # to make it start at year 1
            data,
            color=COLOURS[plotID],
            label=name,
        )
        totals = (name, np.sum(data))

        self.axes.autoscale(enable=True)
        self.axes.relim()
        self.plots[plotID] = plots
        self.totals[plotID] = totals
        self.updateLegend()
        self.updateTotals()
        self.updateShading()
        self.draw()

    def updateShading(self):
        # update vertical limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ylim = (min(ylim[0], -1), max(ylim[1], 1))
        self.axes.set_ylim(ylim)
        upper = [(xlim[0], 0), (xlim[1], 0), (xlim[1], ylim[1]), (xlim[0], ylim[1])]
        lower = [(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], 0), (xlim[0], 0)]
        # shade positive and negative values
        if self.emission_patch == None:
            self.emission_patch = Polygon(
                upper,
                facecolor="r",
                alpha=0.25,
                fill=True,
                edgecolor=None,
                zorder=-1000,
            )
            self.axes.add_patch(self.emission_patch)
        else:
            self.emission_patch.set_xy(upper)
        if self.removal_patch == None:
            self.removal_patch = Polygon(
                lower,
                facecolor="b",
                alpha=0.25,
                fill=True,
                edgecolor=None,
                zorder=-1000,
            )
            self.axes.add_patch(self.removal_patch)
        else:
            self.removal_patch.set_xy(lower)
        # self.removal_patch = None

    def updateTotals(self):
        # No support for ranges (yet)
        # get model names (as keys)
        models = {}
        for c in self.totals:
            models[self.totals[c][0]] = self.totals[c][1:]
        ms = sorted(models.keys())

        years = self.axes.dataLim.get_points()[1, 0]
        outstr = "Total Emissions/Removals\nover %d years " % configuration.N_ACCT
        outstr += "(t CO" + SUB_2 + "e / ha):\n"
        for m in ms:
            outstr += "%s: %.1f\n" % (m, models[m][0])

        # Figure out total of project - baseline for each project
        if len(self.totals) > 1 and 0 in self.totals:
            outstr += "\nNet impact (t CO" + SUB_2 + "e / ha):\n"
            for c in self.totals:
                if c == 0:
                    continue
                outstr += "%s: %.1f\n" % (
                    self.totals[c][0],
                    self.totals[c][1] - self.totals[0][1],
                )

        self.totalsBox.set_text(outstr)

    def removePlot(self, plotID):
        if plotID in self.plots:
            for p in self.plots[plotID]:
                p.remove()
            self.draw()
            del self.totals[plotID]
            del self.plots[plotID]
            if plotID in self.have_range:
                self.have_range.remove(plotID)
            self.updateLegend()
            self.updateTotals()

    def updateLegend(self):
        if len(list(self.plots.keys())) > 0:
            handles, labels = self.axes.get_legend_handles_labels()
            if len(self.have_range) > 0:
                l = Line2D([0, 1], [0, 1], linestyle="--", color="k")
                handles.append(l)
                labels.append("range")
            self.axes.legend(
                handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
            )
        else:
            self.axes.legend_ = None

    def saveFigure(self, fname):
        self.fig.savefig(fname)

    def sizeHint(self):
        w = self.fig.get_figwidth()
        h = self.fig.get_figheight()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(10, 10)
