#!/usr/bin/python

"""Module holding Climate class."""

import logging as log
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import math

from shamba.model import configuration
from shamba.model.common import csv_handler
from shamba.rasters import climate as climate_raster


class Climate(object):

    """
    Object with information about climate of a project.
    Has methods to read, print, and plot climate data.

    Instance variables
    ------------------
    clim    3x12 array (or list) with the following climate data:
    temp    array of temperature in *C for each month
    rain    array of precipitation in mm for each month
    evap    array of evaporation in mm for each month
    
    """
    
    def __init__(self, clim):
        """Initialise climate data.
        
        Args:
            clim: 3x12 array with climate data
        Raises:
            IndexError if the dimensions of clim are not 3x12
        """
        try:
            self.clim = clim
            if self.clim.shape != (3,12):
                raise IndexError
            
            self.clim = clim
            self.temp = self.clim[0]
            self.rain = self.clim[1]
            self.evap = self.clim[2]
            self._sanitize_inputs()

        except IndexError:
            log.exception("Climate data not the right format")
            sys.exit(1)

    @classmethod
    def from_location(cls, location):
        """Construct Climate object using CRU-TS 
        dataset for a given location.
        
        Args: 
            location
        Returns:
            Climate object
        Raises:
            io.FileOpenError if raster file(s) can't be opened/read

        """
        # Location stuff
        lat = location[0]
        long = location[1]
        # Indices for picking out clim data from rasters
        x = math.ceil(180 - 2*lat)
        y = math.ceil(360 + 2)

        # Read data from rasters
        # file path of raster directory 
        folder = os.path.dirname(os.path.abspath(climate_raster.__file__))
        basename = ['tmp_', 'pre_', 'pet_']

        # Populate climate matrix from CRU-TS data
        clim = np.zeros((3,12))
        for k in range(len(basename)):
            for i in range(1,13):
                filename = os.path.join(folder, basename[k], )
                filename += "%d" % i
                filename += ".txt"
                try:
                    clim[k,i-1] = np.loadtxt(
                        filename, usecols=[int(y-1)], skiprows=6+int(x-1))[0]
                except IOError:
                    raise csv_handler.FileOpenError(filename)

        # Account for scaling factor in CRU-TS dataset
        clim *= 0.1

        # Convert pet to mm/month from mm/day
        daysInMonth = np.array([31, 28, 31, 30,
                                31, 30, 31, 31,
                                30, 31, 30, 31])
        clim[2] = clim[2]*daysInMonth
        
        # pet given in CRU-TS 3.1 instead of evaporation, so convert
        clim[2] /= 0.75
    
        return cls(clim)
    
    @classmethod
    def from_csv(
            cls, filename='climate.csv', 
            order=('temp','rain','evap'), 
            isEvap=True):
        """Construct Climate object from a csv file.

        Args:
            filename
            order: tuple of str specifying the column order in csv
            isEvap: whether data has evap or PET - convert to evap
                    if not isEvap
        Returns:
            Climate object
        Raises:
            ValueError: if order doesn't contain 'temp,'rain', and 'evap'
                        in some order
            
        """
        data = csv_handler.read_csv(filename)
        
        try:
            # Create clim array with the correct rows
            clim = np.zeros((3,12))
            correctOrder = ('temp','rain','evap')
            for i in range(3):
                clim[i] = data[:, order.index(correctOrder[i])]
        
            if not isEvap:  # pet given, so convert to evap
                clim[2] /= 0.75
            climate = cls(clim)
        except ValueError:
            log.exception("INCORRECT VALUE(S) IN ORDER ARGUMENT")
            sys.exit(1)
        except IndexError:
            log.exception("Data not in correct format")
            sys.exit(1) 
        
        return climate

    def _sanitize_inputs(self):
        """Check that climate data makes sense."""
        t = self.temp
        r = self.rain
        e = self.evap

        somethingNotRight = False
        for j in range(12):
            if (t[j] < -100.0 or t[j] > 100.0 or np.isnan(t[j])
                or r[j] < 0 or r[j] > 4000.0 or np.isnan(r[j])
                or e[j] < 0 or e[j] > 4000.0 or np.isnan(e[j])
            ):
                somethingNotRight = True
                break
        if somethingNotRight:
            log.warning("Unusual cliamte data. Please check")

    def plot_(self):
        """Plot climate data in a matplotlib figure."""

        xAxis = list(range(1,13))
        fig, ax1 = plt.subplots()
        fig.canvas.manager.set_window_title('Climate data')

        ax1.bar(xAxis, self.rain, align='center', ec='k',fc='w')
        ax1.plot(xAxis, self.evap, 'k--D')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Rain and evaporation (mm/month)')
        ax1.set_title('Monthly Climate Data')

        ax2 = ax1.twinx()
        ax2.plot(xAxis, self.temp, 'b-o')
        ax1.set_xlim(0,13)
        ax2.set_ylabel('Temperature (C)', color='b')

        # Set ax2 to blue to set apart from other axis
        for tl in ax2.get_yticklabels():
            tl.set_color('b')

    def print_(self):
        """Print climate data to stdout."""

        monthNames = ['JAN','FEB','MAR','APR',
                      'MAY','JUN','JUL','AUG',
                      'SEP','OCT','NOV','DEC']

        print ("\nCLIMATE DATA")
        print ("============\n")
        print ("Month   Temp.    Rain     Evap.")
        print ("        (*C)     (mm)     (mm) ")
        print ("-------------------------------")
        for i in range(12):
            print((" %s    %5.2f   %6.2f   %6.2f" % (
                    monthNames[i], self.temp[i], self.rain[i], self.evap[i]
            )))
        print ("")

    def save_(self, file='climate.csv'):
        """Save climate data to a csv file.
        Default path is in cfg.OUT_DIR with filename 'climate.csv'.
        
        Args:
            file: name or path to csv file. If only name is given, file
                  is put in cfg.INP_DIR.

        """
        csv_handler.print_csv(
                file, np.transpose(self.clim),      
                col_names=['temp', 'rain', 'evap']
        )

