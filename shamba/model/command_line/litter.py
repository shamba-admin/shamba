#!/usr/bin/python

"""Module containing litter model."""


import logging as log
import numpy as np

from ..common import csv_handler

from .. import configuration


class LitterModel(object):
    """
    Litter model object. Read litter params
    and calculate residues and inputs to the soil.
    
    Instance variables
    ------------------
    carbon      litter carbon content
    nitrogen    litter nitrogen content
    output      output to soil,fire 
                    (dict with keys 'DMon','DMoff,'carbon','nitrogen')

    """
    
    def __init__(
            self, litter_params, litterFreq, litterQty, litterVector=None):
        """Initialise litter object.
        
        Args: 
            litter_params: dict with litter params 
                           (keys='carbon','nitrogen')
            litterFreq: frequency of litter application
            litterQty: Quantity (in t C ha^-1) of litter when applied.
            litterVector: vector with custom litter additions (t DM / ha)
                          (e.g. for when litter isn't at regular freq.)
                          -> overrides any quantity and freq. info
        Raises:
            KeyError: if litter_params doesn't have the right keys

        """
        
        try:
            self.carbon = litter_params['carbon']
            self.nitrogen = litter_params['nitrogen']
        except KeyError:
            log.exception("Litter parameters not provided.")
            raise
        
        self.output = self.get_inputs(litterFreq, litterQty, litterVector)

    @classmethod
    def from_csv(
            cls, litterFreq, litterQty, 
            filename='litter.csv', row=0,
            litterVector=None):
        """Read litter params from a csv file.
        
        Args: 
            filename
            row: which row to read from file
            other same as __init__
        Returns:
            Litter object
        Raises:
            IndexError: if row > number of rows in csv,
                        or csv doesn't have 2 columns

        """

        data = csv_handler.read_csv(filename)
        data = np.atleast_2d(data)
        try:
            params = {
                    'carbon': data[row,0],
                    'nitrogen': data[row,1]
            }
            litter = cls(params, litterFreq, litterQty, litterVector)
        except IndexError:
            log.exception("Can't find row %d in %s" % (row, filename))
            raise

        return litter

    @classmethod
    def from_defaults(cls, litterFreq, litterQty, litterVector=None):
        """Construct litter object from default parameters.
        
        Args: 
            same as __init__
        
        """
        
        # Carbon and nitrogen content of litter input defaults
        params = {
                'carbon': 0.5,
                'nitrogen': 0.018
        }
        return cls(params, litterFreq, litterQty, litterVector)

    @classmethod
    def synthetic_fert(cls, freq, qty, nitrogen, vector=None):
        """Synthetic fertiliser (special case of litter).
        Be sure to keep separate though when passing a litter object to
        other methods/classes. (e.g. fert isn't an input to soil model)""" 
        params = {
                'carbon': 0,
                'nitrogen': nitrogen
        }

        return cls(params, freq, qty, vector)

    def get_inputs(self, litterFreq, litterQty, litterVector):
        """Calculate and return DM, C, and N inputs to 
        soil from additional litter.
        
        Args:
            litterFreq: frequency of litter addition
            litterQty: amount of dry matter added to field 
                       when litter added in t DM ha^-1
        Returns:
            output: dict with soil,fire inputs due to litter
                    (keys='carbon','nitrogen','DMon','DMoff')

        """
        
        if litterVector is None:
            # Construct vectors for DM, C, N
            Cinput = np.zeros(configuration.N_YEARS)
            Ninput = np.zeros(configuration.N_YEARS)
            DMinput = np.zeros(configuration.N_YEARS)
            
            # loop through years when litter is added                   
            if litterFreq == 0:
                years = 0
            else:
                years = list(range(-1,configuration.N_YEARS,litterFreq))
                years = years[1:]
            DMinput[years] = litterQty
            Cinput[years] = litterQty * self.carbon
            Ninput[years] = litterQty * self.nitrogen 
        else:
            # DM vector already specified
            DMinput = np.array(litterVector)
            Cinput = DMinput * self.carbon
            Ninput = DMinput * self.nitrogen

        # Standard output (same as crop and tree classes)
        output = {}
        output['above'] = {
                'carbon': Cinput,
                'nitrogen': Ninput,
                'DMon': DMinput,
                'DMoff': np.zeros(len(Cinput))
        }
        output['below'] = {
                'carbon': np.zeros(len(Cinput)),
                'nitrogen': np.zeros(len(Cinput)),
                'DMon': np.zeros(len(Cinput)),
                'DMoff': np.zeros(len(Cinput))
        }
        
        return output
    
    def save_(self, file='litter_model.csv'):
        """Save output to a csv. Default path is OUTPUT_DIR

        Args:
            file: name or path to csv
        
        """
        cols = []
        data = []
        for s1 in ['above', 'below']:
            for s2 in ['carbon', 'nitrogen', 'DMon', 'DMoff']:
                cols.append(s2+"_"+s1)
                data.append(self.output[s1][s2])
        data = np.column_stack(tuple(data))
        csv_handler.print_csv(file, data, col_names=cols)
