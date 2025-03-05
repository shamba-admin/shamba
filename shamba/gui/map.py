# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:09:08 2012

@author: mhagdorn
"""

import matplotlib.image
import numpy
import pylab
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from PyQt5 import QtCore, QtGui, QtWidgets


class MapWidget(FigureCanvas):
    """a QtWidget and FigureCanvasAgg that displays a map"""

    def __init__(
        self, parent=None, name=None, width=5, height=4, dpi=100, bgcolor=None
    ):
        self.fig = Figure(
            figsize=(width, height), dpi=dpi, facecolor=bgcolor, edgecolor=bgcolor
        )
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.updateGeometry()

        self.mpl_connect("button_press_event", self.onpick)

        self.m = None
        self.point = None
        self.pickCallback = None

    def onpick(self, event):
        if event.inaxes != None and self.pickCallback != None:
            self.pickCallback(self.m(event.xdata, event.ydata, inverse=True))

    def drawBasemap(self, basemapData):
        if basemapData.projection == "tmerc":
            self.m = Basemap(
                llcrnrlon=basemapData.llcrnrlon,
                llcrnrlat=basemapData.llcrnrlat,
                urcrnrlon=basemapData.urcrnrlon,
                urcrnrlat=basemapData.urcrnrlat,
                resolution="i",
                projection=basemapData.projection,
                lon_0=basemapData.lon_0,
                lat_0=basemapData.lat_0,
                ax=self.axes,
            )
        else:
            raise ValueError("Unknown Projection %s" % basemapData.projection)

        if basemapData.geotiff == None:
            self.m.fillcontinents(color="coral", lake_color="aqua")
            self.m.drawmapboundary(fill_color="aqua")
        else:
            img = matplotlib.image.imread(basemapData.geotiff)
            self.m.imshow(img)

        self.m.drawcoastlines()
        # draw parallels and meridians.
        self.m.drawcountries()
        self.m.drawparallels(
            numpy.arange(
                basemapData.first_parallel,
                basemapData.last_parallel + basemapData.delta_parallel,
                basemapData.delta_parallel,
            ),
            labels=[False, True, True, False],
        )
        self.m.drawmeridians(
            numpy.arange(
                basemapData.first_meridian,
                basemapData.last_meridian + basemapData.delta_meridian,
                basemapData.delta_meridian,
            ),
            labels=[True, False, False, True],
        )

        x, y = self.m(
            [
                basemapData.llcrnrlon
                + (basemapData.urcrnrlon - basemapData.llcrnrlon) / 2.0
            ],
            [
                basemapData.llcrnrlat
                + (basemapData.urcrnrlat - basemapData.llcrnrlat) / 2.0
            ],
        )

    def plotPoint(self, lon, lat):
        if self.m != None:
            x, y = self.m(lon, lat)
            if self.point != None:
                self.point.set_data(x, y)
            else:
                self.point = self.m.plot(x, y, marker="o", color="k")[0]
            self.draw()

    def plotLocations(self, locations):
        for loc in locations.locations:
            x, y = self.m(loc[0], loc[1])
            self.m.plot(x, y, marker="o", color="r")

    def sizeHint(self):
        w = self.fig.get_figwidth()
        h = self.fig.get_figheight()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(10, 10)

    def saveFigure(self, fname):
        self.fig.savefig(fname)
