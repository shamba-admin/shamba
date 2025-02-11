#!/usr/bin/python


import os
import sys
import logging as log
import numpy as np
from shamba.model import configuration
from shamba.model.common import csv_handler


# --------------------------
# Read species data from csv
# --------------------------

SPP_LIST = [
        'grains',
        'beans and pulses',
        'tubers',
        'root crops',
        'n-fixing forages',
        'non-n-fixing forages',
        'perennial grasses',
        'grass-clover mixtures',
        'maize',
        'wheat',
        'winter wheat',
        'spring wheat',
        'rice',
        'barley',
        'oats',
        'millet',
        'sorghum',
        'rye',
        'soyabean',
        'dry bean',
        'potato',
        'peanut',
        'alfalfa',
        'non-legume hay'
]

# Read csv file with default crop data
CROP_SPP = {}
_data = csv_handler.read_csv('crop_ipcc_defaults.csv', cols=(2,3,4,5,6,7,8))
_data = np.atleast_2d(_data)

_slope = _data[:,0]
_intercept = _data[:,1]
_nitrogenBelow = _data[:,2]
_nitrogenAbove = _data[:,3]
_carbonBelow = _data[:,4]
_carbonAbove = _data[:,5]
_rootToShoot = _data[:,6]

for _i in range(len(SPP_LIST)):
    _spp = SPP_LIST[_i]
    CROP_SPP[_spp] = {
            'species': _spp,
            'slope': _slope[_i],
            'intercept': _intercept[_i],
            'nitrogenBelow': _nitrogenBelow[_i],
            'nitrogenAbove': _nitrogenAbove[_i],
            'carbonBelow': _carbonBelow[_i],
            'carbonAbove': _carbonAbove[_i],
            'rootToShoot': _rootToShoot[_i]
    }


class CropParams(object):
   
    """
    Crop object to hold crop params. Can be initialised from species name,
    species index, csv file, or manually (callling __init__ with params
    in a dict)

    Instance variables
    ------------------
    species         crop species
    slope           crop IPCC slope
    intercept       crop IPCC y-intercept
    nitrogenBelow   crop below-ground nitrogen content as a fraction
    nitrogenAbove   crop above-ground nitrogen content as a fraction
    carbonBelow     crop below-ground carbon content as a fraction
    carbonAbove     crop above-ground carbon content as a fraction
    rootToShoot     crop root-to-shoot ratio

    """ 

    ROOT_IN_TOP_30 = 0.7
    def __init__(self, crop_params):
        """Initialise Crop object which holds crop parameters.

        Args:
            crop_params: dict with crop params
                         keys='species','slope','intercept',
                              'nitrogenBelow','nitrogenAbove',
                              'carbonAbove','carbonBelow','rootToShoot'
        Raises:
            KeyError: if crop_params doesn't have the right keys

        """
        try:
            self.species = crop_params['species']
            self.slope = crop_params['slope']
            self.intercept = crop_params['intercept']
            self.nitrogenBelow = crop_params['nitrogenBelow']
            self.nitrogenAbove = crop_params['nitrogenAbove']
            self.carbonBelow = crop_params['carbonBelow']
            self.carbonAbove = crop_params['carbonAbove']
            self.rootToShoot = crop_params['rootToShoot']
        except KeyError:
            log.exception("CROP PARAMETERS NOT PROVIDED")
            sys.exit(1)

    @classmethod
    def from_species_name(cls, species):
        """Construct Crop object from species default in CROP_SPP.

        Args: 
            species: species name to be read from speciesList
        Returns:
            Crop object
        Raises:
            KeyError: if species isn't a key in the cs.crop dict

        """
        species = species.lower()
        try:
            crop = cls(CROP_SPP[species])
        except KeyError:
            log.exception(
                "COULD NOT FIND SPECIES DATA IN DEFAULTS FOR %s" % species
            )
            sys.exit(1)

        return crop

    @classmethod
    def from_species_index(cls, index):
        """Construct Crop object from index of species in the csv
        
        Args:
            speciesNum: index of the species
                        (1-indexed, so off by one from index in SPP_LIST)
        Return:
            Crop object
        Raises:
            IndexError: if speciesNum isn't a valid index in species list

        """
        try:
            index = int(index)
            # csv list is 1-indexed
            species = SPP_LIST[index-1]
            crop = cls(CROP_SPP[species])
        except IndexError:
            log.exception(
                    "COULD NOT FIND SPECIES DATA CORRESPONDING " + \
                     "TO SPECIES NUMBER %d" % index
            )
            sys.exit(1)

        return crop 

    @classmethod
    def from_csv(cls, speciesName, filename, row=0):
        """Construct Crop object using data from a csv which 
        is structured like the master csv (crop_ipcc_defaults.csv).
        
        Args: 
            speciesName: name of species (can be anything)
            filename: filename of csv with species info
            row: row in the csv to be read (0-indexed)
        Returns:    
            Crop object
        Raises:
            IndexError: if row > number of rows in csv,
                        or csv doesn't have 7 columns

        """

        data = csv_handler.read_csv(filename, cols=(2,3,4,5,6,7,8))
        data = np.atleast_2d(data) # account for when only one row in file

        try:
            params = {
                    'species': speciesName,
                    'slope': data[row,0],
                    'intercept': data[row,1],
                    'nitrogenBelow': data[row,2],
                    'nitrogenAbove': data[row,3],
                    'carbonBelow': data[row,4],
                    'carbonAbove': data[row,5],
                    'rootToShoot': data[row,6]
            }
            crop = cls(params)
        except IndexError:
            log.exception("CAN'T FIND ROW %d IN %s" % (row,filename))
            sys.exit(1)

        return crop

    def save_(self, file='crop_params.csv'):
        """Save crop params in a csv. 
        Default path is in OUT_DIR with filename 'crop_params.csv'

        Args:
            file: name or path to csv file

        """
        # Index is 0 if not in the SPP_LIST
        if self.species in SPP_LIST:
            index = SPP_LIST.index(self.species) + 1
        else:
            index = 0

        data = [
                index, self.species, self.slope, self.intercept, 
                self.nitrogenBelow, self.nitrogenAbove, self.carbonBelow, 
                self.carbonAbove, self.rootToShoot
        ]
        cols = [
                'Sc','Name','a', 'b', 'crop_bgn', 'crop_agn',
                'crop_bgc', 'crop_agc', 'crop_rs'
        ]
        csv_handler.print_csv(file, data, col_names=cols)

