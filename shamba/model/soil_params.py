#!/usr/bin/python

"""Module containing Soil class."""

import logging as log
import os
import sys

import numpy as np
# import gdal, gdalconst

from shamba.model import io_, cfg
from shamba.rasters import soil as soil_raster


class SoilParams(object):
   
    """
    Object to hold soil parameter information.
    
    Instance variables
    ------------------
    Cy0     soil carbon at start of project in t C ha^-1 (year 0)
    clay    soil clay content as percentage
    depth   depth of soil in cm (default=30)
    Ceq     soil carbon at equilibrium in t C ha^-1 (calculated from Cy0)
    iom     soil inert organic matter in t C ha^-1 (calculated from Ceq)

    """

    def __init__(self, soil_params):
        """Initialise soil data.
        
        Args: 
            soil_params: dict with soil params (keys='Cy0,'clay')
        Raises: 
            KeyError: if dict doesn't have the right keys
            TypeError: if non-numeric type is used for soil_params

        """
        try:
            self.Cy0 = soil_params['Cy0']
            self.clay = soil_params['clay']
            SoilParams._sanitize_params(soil_params)
        except KeyError:
            log.exception("Soil parameters not in right format")
            sys.exit(1)
        
        try:
            self.depth = 30.0
            self.Ceq = 1.25 * self.Cy0
            self.iom = 0.049 * self.Ceq ** 1.139
        except TypeError:
            log.exception("Soil parameters not of the right type")
            sys.exit(1)

    @classmethod
    def from_csv(cls, filename='soil.csv'):
        """Construct Soil object from csv data."""
        data = io_.read_csv(filename)

        params = {
                'Cy0': data[0],
                'clay': data[1]
        }
        return cls(params)

    @classmethod
    def from_location(cls, location):
        """Construct Soil object from HWSD data for given location
        
        Args:
            location: location to look for in HWSD
        Returns:
            Soil object

        """
        mu_global = cls._get_identifier(location)
        Cy0, clay = cls._get_data_from_identifier(mu_global)
        params = {
                'Cy0': Cy0,
                'clay': clay
        }
        return cls(params)
    
    @classmethod
    def _sanitize_params(cls, params):
        """Check that provided soil data makes sense.

        Args:
            params: soil params dict with keys 'Cy0' and 'clay'
        """
        
        Cy0 = params['Cy0']
        clay = params['clay']

        if (
                clay < 0 or clay > 100
                or Cy0 < 0 or Cy0 > 10000
        ):
            log.warning("Unusual soil parameters. Please check.")
        
    def print_(self):
        """Print soil information to stdout."""
        print ("\nSOIL INFORMATION")
        print ("================\n")
        print ("Equilibrium C -",self.Ceq)
        print ("C at y=0  - - -",self.Cy0)
        print ("IOM - - - - - -",self.iom)
        print ("Clay  - - - - -",self.clay)
        print ("Depth - - - - -",self.depth)         
        print ("")

    def save_(self, file='soil_params.csv'):
        """Save soil params to a csv file. Default path is in OUT_DIR
        with filename soil_params.csv

        Args:
            file: name or path to csv file. If path is not given
                  (only name), put in OUT_DIR for this program run.

        """
        data = np.array(
                [self.Cy0, self.clay, self.Ceq, self.iom, self.depth])
        cols = ['Cy0', 'clay', 'Ceq', 'iom', 'depth']
        io_.print_csv(file, data, col_names=cols)

    @staticmethod
    def _get_identifier(location):
        """Find MU_GLOBAL for given location from the HWSD .bil raster."""

        y = location[0] # lat
        x = location[1] # long

        # gdal setup
        gdal.AllRegister()
        driver = gdal.GetDriverByName('HFA')
        driver.Register()
        gdal.UseExceptions()

        # open file
        try:
            filename = os.path.join(
                    os.path.dirname(os.path.abspath(soil_raster.__file__)),
                    'hwsd.bil'
            )
            ds = gdal.Open(filename)
        except RuntimeError:
            raise io_.FileOpenError(filename)

        cols = ds.RasterXSize
        rows = ds.RasterYSize
        bands = ds.RasterCount

        # georeference info
        transform = ds.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        width = transform[1]    # x resolution
        height = transform[5]   # y resolution

        # FIND VALUES
        # cast as ints
        xInt = int((x - xOrigin) / width)
        yInt = int((y - yOrigin) / height)
        band = ds.GetRasterBand(1)  # one-indexed

        data = band.ReadAsArray(xInt, yInt, 1, 1)
        value = data[0,0]   # MU_GLOBAL for input to HWSD_data.csv

        return value

    @staticmethod
    def _get_data_from_identifier(mu):
        """Get soil data from csv given MU_GLOBAL from the raster."""

        filename = os.path.join(
                os.path.dirname(os.path.abspath(soil_raster.__file__)),
                'HWSD_data.csv'
        )

        soilTable = io_.read_mixed_csv(
                filename,
                cols=(0,1,2,3,4,5,6,7,8,9,10,11,12),
                types=(int, int, float, int, "|S25", float, float,
                        float, "|S15", float, float, float, float)
        )

        # Find rows with mu in the MU_GLOBAL column (column 1)
        muRows = [] # rows we want as list of records
        for row in soilTable:
            if row[1] == mu:
                muRows.append(row)
        if not muRows:
            log.warning("COULD NOT FIND %d IN HWSD_DATA.csv", mu)
            return None

        # Weighted sum of SOC and clay
        cy0 = 0
        clay = 0
        for row in muRows:
            # weighted sum of SOC and clay
            cy0 += row[12] * row[2]
            clay += row[7] * row[2]
        cy0 /= 100         # account for percentages
        clay /= 100

        return cy0, clay
   

if __name__ == '__main__':
    data = io_.read_csv(
            os.path.join(
                cfg.SHAMBA_DIR,'rasters','soil','muGlobalTestValues.csv')
    )
    long = data[:,1]
    lat = data[:,2]
    mu = data[:,3]

    for i in range(len(mu)):
        a = Soil((lat[i],long[i]))
        print ("\nlocation = %f, %f " % (lat[i], long[i]))
        print ("mu actual= %d" % mu[i])
        print ("mu python= %d" % a.mu_global)

