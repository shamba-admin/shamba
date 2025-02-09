#!/usr/bin/python

"""Module holding Crop class for the crop model."""

import logging as log
import sys
import numpy as np

from . import configuration, io_
from .crop_params import CropParams


class CropModel(object):

    """
    Crop model object. Calculate residues and soil inputs 
    for given parameters.

    Instance variables
    ------------------
    crop_params     CropParams object with crop params (slope, carbon, etc.)
    output          output to soil,fire in t C ha^-1
                        (dict with keys 'carbon','nitrogen','DMon','DMoff')

    """

    def __init__(self, crop_params, cropYield, leftInField):
        """Initialise crop object.
        
        Args: 
            crop_params: CropParams object with crop params
            cropYield: dry matter yield of the crop in t C ha^-1
            leftInField: fraction of residues left in field post-harvest

        """
        
        self.crop_params = crop_params
        self.output = self.get_inputs(cropYield, leftInField)

    def get_inputs(self, cropYield, leftInField):
        """Calculate and return soil carbon inputs, nitrogen inputs,
        on-farm residues, and off-farm residues from soil parameters. 
        
        Use the IPCC slope and intercept values
        outlined in 2006 national GHG assessment guidelines (table 11.2)
        to model

        Args:
            cropYield: yearly dry-matter crop yield (in t C ha^-1)
            leftInField: fraction left in field after harvest
        Returns:
            output: dict with soil,fire inputs due to crop
                     (keys='carbon','nitrogen','DMon','DMoff')
        """

        # residues
        res = cropYield*self.crop_params.slope + self.crop_params.intercept
        res *= np.ones(configuration.N_YEARS)  # convert to array 
        resAG = res*leftInField
        resBG = cropYield + res
        resBG *= self.crop_params.rootToShoot * CropParams.ROOT_IN_TOP_30

        output = {}
        
        # Standard outputs - in tonnes of carbon and as vectors
        output['above'] = {
                'carbon': resAG * self.crop_params.carbonAbove,
                'nitrogen': resAG * self.crop_params.nitrogenAbove,
                'DMon': resAG,
                'DMoff': resAG * (1 - leftInField)
        }
        output['below'] = {
                'carbon': resBG * self.crop_params.carbonBelow,
                'nitrogen': resBG * self.crop_params.nitrogenBelow,
                'DMon': resBG,
                'DMoff': np.zeros(len(res))
        }

        return output

    def save_(self, file='crop_model.csv'):
        """Save output of crop model to a csv file.
        Default path is in OUT_DIR.

        Args:
            file: name or path to csv file

        """
        cols = []
        data = []
        for s1 in ['above', 'below']:
            for s2 in ['carbon','nitrogen','DMon','DMoff']:
                cols.append(s2+"_"+s1)
                data.append(self.output[s1][s2])
        data = np.column_stack(tuple(data))
        io_.print_csv(file, data, col_names=cols)

