#!/usr/bin/python

import logging as log
import sys

import numpy as np
from marshmallow import Schema, fields, post_load
from model.common import csv_handler


# ----------------------------------
# Read species data from csv
# (run when this module is imported)
# ----------------------------------
# abridged species list
SPP_LIST = [1, 2, 3]

TREE_SPP = {}
# For GUI, the filename is 'tree_defaults.csv'
_data = csv_handler.read_csv("tree_defaults_cl.csv", cols=(2, 3, 4, 5, 6, 7, 8, 9))
_data = np.atleast_2d(_data)
_nitrogen = np.zeros((len(SPP_LIST), 5))
for _i in range(len(SPP_LIST)):
    _nitrogen[_i] = np.array(
        [_data[_i, 0], _data[_i, 1], _data[_i, 2], _data[_i, 3], _data[_i, 4]]
    )

_carbon = _data[:, 5]
_rootToShoot = _data[:, 6]
_dens = _data[:, 7]

for _i in range(len(SPP_LIST)):
    _spp = SPP_LIST[_i]
    TREE_SPP[_spp] = {
        "species": _spp,
        "dens": _dens[_i],
        "carbon": _carbon[_i],
        "nitrogen": _nitrogen[_i],
        "rootToShoot": _rootToShoot[_i],
    }


# -------------------------------------------------------
# Tree object, holding info for a particular type of tree
# -------------------------------------------------------

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

class TreeParamsData:
    def __init__(
        self,
        species,
        dens,
        nitrogen,
        carbon,
        root_to_shoot,
    ):
        self.species = species
        self.dens = dens
        self.nitrogen = nitrogen
        self.carbon = carbon
        self.root_to_shoot = root_to_shoot

def validate_species(value):
    # Determining whether the value can be interpreted as a string or an integer
    errors = (
        [f"{value} must be convertible to a string."] * (not isinstance(str(value), str)) +
        [f"{value} must be convertible to an integer."] * (not value.isdigit() if isinstance(value, str) else False)
    )
    return errors

class TreeParamsSchema(Schema):
    species = fields.Raw(required=True, validate=lambda v: validate_species(v))
    dens = fields.Float(required=True)
    carbon = fields.Float(required=True)
    nitrogen = fields.List(fields.Float, required=True)
    root_to_shoot = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeParamsData(**data)

def create(tree_params):
    """Initialise tree data.

    Args:
        tree_params: dict with tree params
                        keys='species','dens','carbon',
                            'nitrogen','rootToShoot'
    Raises:
        KeyError: if tree_params doesn't have the right keys

    """
    params = {
        "species": tree_params["species"],
        "dens": tree_params["dens"],
        "carbon": tree_params["carbon"],
        "nitrogen": tree_params["nitrogen"],
        "root_to_shoot": tree_params["rootToShoot"]
    }

    schema = TreeParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree params: {errors}")

    return schema.load(params)

def from_species_name(species):
    """Construct Tree object from species defaults in tree_spp dict.

    Args:
        species: species name that's a key in tree_spp
    Returns:
        Tree object
    Raises:
        KeyError: is species isn't a key in the tree_spp dict

    """
    return create(TREE_SPP[species])

def from_species_index(index):
    """Construct Tree object from index of species in the csv.

    Args:
        index: index of the species (1-indexed, so off by one from
                SPP_LIST index
    Returns:
        Tree object
    Raises:
        IndexError: if index is not a valid index in the species list

    """
    index = int(index)
    species = SPP_LIST[index - 1]

    return create(TREE_SPP[species])

    # TODO: is this being used?
    # def _repr(self):
    #     return "_repr_"

def from_csv(speciesName, filename, row=0):
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

    data = csv_handler.read_csv(filename, cols=(2, 3, 4, 5, 6, 7, 8, 9))
    data = np.atleast_2d(data)  # to account for if there's only one row

    params = {
        "species": speciesName,
        "nitrogen": np.array(
            [
                data[row, 0],
                data[row, 1],
                data[row, 2],
                data[row, 3],
                data[row, 4],
            ]
        ),
        "carbon": data[row, 5],
        "rootToShoot": data[row, 6],
        "dens": data[row, 7],
    }
    return create(params)

def save(tree_params, file="tree_params.csv"):
    """Save tree params to a csv.
    Default path is in OUTPUT_DIR with filename 'tree_params.csv'

    Args:
        file: name or path to csv file

    """
    # index is 0 if not in the SPP_LIST
    if tree_params.species in SPP_LIST:
        index = SPP_LIST.index(tree_params.species) + 1
    else:
        index = 0

    data = [
        index,
        tree_params.species,
        tree_params.nitrogen[0],
        tree_params.nitrogen[1],
        tree_params.nitrogen[2],
        tree_params.nitrogen[3],
        tree_params.nitrogen[4],
        tree_params.carbon,
        tree_params.root_to_shoot,
        tree_params.dens,
    ]
    cols = [
        "Sc",
        "Name",
        "N_leaf",
        "N_branch",
        "N_stem",
        "N_croot",
        "N_froot",
        "C",
        "rw",
        "dens",
    ]
    csv_handler.print_csv(file, data, col_names=cols)
