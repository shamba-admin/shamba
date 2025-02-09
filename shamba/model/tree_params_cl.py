#!/usr/bin/python

import logging as log
import os
import sys
import pdb

import numpy as np
from shamba.model import configuration, io_


# ----------------------------------
# Read species data from csv
# (run when this module is imported)
# ---------------------------------- 
# abridged species list
SPP_LIST = [
        1, 2, 3
]

TREE_SPP = {}
_data = io_.read_csv('tree_defaults_cl.csv',cols=(2,3,4,5,6,7,8,9))
_data = np.atleast_2d(_data)
_nitrogen = np.zeros((len(SPP_LIST),5))
for _i in range(len(SPP_LIST)):
    _nitrogen[_i] = np.array(
            [_data[_i,0],_data[_i,1],_data[_i,2],_data[_i,3],_data[_i,4]]
    )

_carbon = _data[:,5]
_rootToShoot = _data[:,6]
_dens = _data[:,7]

for _i in range(len(SPP_LIST)):
    _spp = SPP_LIST[_i]
    TREE_SPP[_spp] = {
            'species': _spp,
            'dens': _dens[_i],
            'carbon': _carbon[_i],
            'nitrogen': _nitrogen[_i],
            'rootToShoot': _rootToShoot[_i]
    }


# -------------------------------------------------------
# Tree object, holding info for a particular type of tree
# -------------------------------------------------------

class TreeParams(object):

    """
    Tree object to hold params. Can be initialised from species name,
    species index, csv, or manually (calling __init__ with params
    in a dict).
    
    Instance variables
    ----------------
    species         tree species name
    dens        tree density in g cm^-3
    carbon          tree carbon content as a fraction
    nitrogen        tree nitrogen content as a fraction 
    rootToShoot     tree root-to-shoot ratio 

    """
    
    ROOT_IN_TOP_30 = 0.7
    def __init__(self, tree_params):
        """Initialise tree data.

        Args:
            tree_params: dict with tree params
                         keys='species','dens','carbon',
                              'nitrogen','rootToShoot'
        Raises:
            KeyError: if tree_params doesn't have the right keys

        """
        try:
            self.species = tree_params['species']
            self.dens = tree_params['dens']
            self.carbon = tree_params['carbon']
            self.nitrogen = tree_params['nitrogen']
            self.rootToShoot = tree_params['rootToShoot']
        except KeyError:
            log.exception("Tree parameters not provided")
            sys.exit(1)

    @classmethod
    def from_species_name(cls, species):
        """Construct Tree object from species defaults in tree_spp dict.

        Args:
            species: species name that's a key in tree_spp
        Returns:
            Tree object
        Raises:
            KeyError: is species isn't a key in the tree_spp dict

        """
        try:
            tree = cls(TREE_SPP[species])
        except KeyError:
            log.exception(
                "Could not find species data in defaults for %s" % species)
            sys.exit(1)

        return tree

    @classmethod
    def from_species_index(cls, index):
        """Construct Tree object from index of species in the csv.

        Args:
            index: index of the species (1-indexed, so off by one from 
                   SPP_LIST index
        Returns:
            Tree object
        Raises:
            IndexError: if index is not a valid index in the species list
        
        """
        try:
            index = int(index)
            species = SPP_LIST[index-1]
            tree = cls(TREE_SPP[species])
        except IndexError:
            log.exception(
                    "Could not find species data corresponding to " + \
                    "species number %d" % index)
            sys.exit(1)
        
        return tree
        
        def _repr(self):
            return '_repr_'

    @classmethod
    def from_csv(cls, speciesName, filename, row=0):
        """Construct Tree object using data from a csv which
        is structured like the master csv (tree_defaults.csv)

        Args:
            speciesName: name of species (can be anything)
            filename: csv file with the info
            row: row in the csv to be read (0-indexed)
        Returns:
            Tree object
        Raises:
            IndexError: if row > # rows in csv, or csv doesn't have 8 cols
        
        """

        
        data = io_.read_csv(filename, cols=(2,3,4,5,6,7,8,9))
        data = np.atleast_2d(data)  # to account for if there's only one row

        try:
            params = {
                    'species': speciesName,
                    'nitrogen': np.array([data[row,0],
                                          data[row,1],
                                          data[row,2],
                                          data[row,3],
                                          data[row,4]
                                ]),
                    'carbon': data[row,5],
                    'rootToShoot': data[row,6],
                    'dens': data[row,7]
            }
            tree = cls(params)
        
        except IndexError:
            log.exception("Couldn't find row %d in %s" % (row, filename))
            sys.exit(1)

        return tree

    def save_(self, file='tree_params.csv'):
        """Save tree params to a csv. 
        Default path is in OUT_DIR with filename 'tree_params.csv'

        Args:
            file: name or path to csv file

        """
        # index is 0 if not in the SPP_LIST
        if self.species in SPP_LIST:
            index = SPP_LIST.index(self.species) + 1
        else:
            index = 0

        data = [
                index, self.species, self.nitrogen[0], self.nitrogen[1],
                self.nitrogen[2], self.nitrogen[3], self.nitrogen[4],
                self.carbon, self.rootToShoot, self.dens
        ]
        cols = ['Sc', 'Name', 'N_leaf', 'N_branch', 'N_stem', 'N_croot',
                'N_froot', 'C', 'rw', 'dens']
        io_.print_csv(file, data, col_names=cols)

