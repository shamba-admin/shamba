#!/usr/bin/python


import logging as log
import sys

import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler
import model.common.constants as CONSTANTS

# --------------------------
# Read species data from csv
# --------------------------

SPP_LIST = [
    "grains",
    "beans and pulses",
    "tubers",
    "root crops",
    "n-fixing forages",
    "non-n-fixing forages",
    "perennial grasses",
    "grass-clover mixtures",
    "maize",
    "wheat",
    "winter wheat",
    "spring wheat",
    "rice",
    "barley",
    "oats",
    "millet",
    "sorghum",
    "rye",
    "soyabean",
    "dry bean",
    "potato",
    "peanut",
    "alfalfa",
    "non-legume hay",
]

# Read csv file with default crop data
def load_crop_species_data(
    filename: str = "crop_params.csv",
) -> dict[str, dict]:
    """
    Load crop species data from CSV file.
    
    Args:
        filename: Name of the CSV file to load (default: "crop_params.csv")
        
    Returns:
        Dictionary mapping species names to their parameter dictionaries
    """
    data = csv_handler.read_csv(filename, cols=(2, 3, 4, 5, 6, 7, 8))
    data = np.atleast_2d(data)
    
    slope = data[:, 0]
    intercept = data[:, 1]
    nitrogenBelow = data[:, 2]
    nitrogenAbove = data[:, 3]
    carbonBelow = data[:, 4]
    carbonAbove = data[:, 5]
    rootToShoot = data[:, 6]
    
    return {
        spp: {
            "species": spp,
            "slope": slope[i],
            "intercept": intercept[i],
            "nitrogen_below": nitrogenBelow[i],
            "nitrogen_above": nitrogenAbove[i],
            "carbon_below": carbonBelow[i],
            "carbon_above": carbonAbove[i],
            "root_to_shoot": rootToShoot[i],
        }
        for i, spp in enumerate(SPP_LIST)
    }


class CropParamsData:
    """
    Crop object to hold crop params. Can be initialised from species name,
    species index, csv file, or manually (callling __init__ with params
    in a dict)

    Instance variables
    ------------------
    species         crop species
    slope           crop IPCC slope
    intercept       crop IPCC y-intercept
    nitrogen_below   crop below-ground nitrogen content as a fraction
    nitrogen_above   crop above-ground nitrogen content as a fraction
    carbon_below     crop below-ground carbon content as a fraction
    carbon_above     crop above-ground carbon content as a fraction
    root_to_shoot     crop root-to-shoot ratio

    """

    def __init__(
        self,
        species,
        slope,
        intercept,
        nitrogen_below,
        nitrogen_above,
        carbon_below,
        carbon_above,
        root_to_shoot,
    ):
        self.species = species
        self.slope = slope
        self.intercept = intercept
        self.nitrogen_below = nitrogen_below
        self.nitrogen_above = nitrogen_above
        self.carbon_below = carbon_below
        self.carbon_above = carbon_above
        self.root_to_shoot = root_to_shoot


class CropParamsSchema(Schema):
    species = fields.String(required=True)
    slope = fields.Float(required=True)
    intercept = fields.Float(required=True)
    nitrogen_below = fields.Float(required=True)
    nitrogen_above = fields.Float(required=True)
    carbon_below = fields.Float(required=True)
    carbon_above = fields.Float(required=True)
    root_to_shoot = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return CropParamsData(**data)


def from_species_name(species) -> CropParamsData:
    """Construct Crop object from species default in CROP_SPP.

    Args:
        species: species name to be read from species list
    Returns:
        Crop object
    Raises:
        KeyError: if species isn't a key in the cs.crop dict

    """
    species = species.lower()
    CROP_SPP = load_crop_species_data()
    crop_params = CROP_SPP[species]
    params = {
        "species": crop_params["species"],
        "slope": crop_params["slope"],
        "intercept": crop_params["intercept"],
        "nitrogen_below": crop_params["nitrogen_below"],
        "nitrogen_above": crop_params["nitrogen_above"],
        "carbon_below": crop_params["carbon_below"],
        "carbon_above": crop_params["carbon_above"],
        "root_to_shoot": crop_params["root_to_shoot"],
    }

    schema = CropParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in crop params: {errors}")

    return schema.load(params)  # type: ignore


def from_species_index(index) -> CropParamsData:
    """Construct Crop object from index of species in the csv

    Args:
        index: index of the species
                    (1-indexed, so off by one from index in SPP_LIST)
    Return:
        Crop object

    """
    index = int(index)
    # csv list is 1-indexed
    species = SPP_LIST[index - 1]
    CROP_SPP = load_crop_species_data()
    crop_params = CROP_SPP[species]
    params = {
        "species": crop_params["species"],
        "slope": crop_params["slope"],
        "intercept": crop_params["intercept"],
        "nitrogen_below": crop_params["nitrogen_below"],
        "nitrogen_above": crop_params["nitrogen_above"],
        "carbon_below": crop_params["carbon_below"],
        "carbon_above": crop_params["carbon_above"],
        "root_to_shoot": crop_params["root_to_shoot"],
    }
    schema = CropParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in crop params data: {str(errors)}")
        print(
            "COULD NOT FIND SPECIES DATA CORRESPONDING "
            + "TO SPECIES NUMBER %d" % index
        )

    return schema.load(params)  # type: ignore


def from_csv(species_name, filename, row=0) -> CropParamsData:
    """Construct Crop object using data from a csv which
    is structured like the master csv (crop_params_defaults.csv).

    Args:
        species_name: name of species (can be anything)
        filename: filename of csv with species info
        row: row in the csv to be read (0-indexed)
    Returns:
        Crop object
    """

    data = csv_handler.read_csv(filename, cols=(2, 3, 4, 5, 6, 7, 8))
    data = np.atleast_2d(data)  # account for when only one row in file

    params = {
        "species": species_name,
        "slope": data[row, 0],
        "intercept": data[row, 1],
        "nitrogen_below": data[row, 2],
        "nitrogen_above": data[row, 3],
        "carbon_below": data[row, 4],
        "carbon_above": data[row, 5],
        "root_to_shoot": data[row, 6],
    }
    schema = CropParamsSchema()
    errors = schema.validate(params)

    print(f"Errors in crop params: {errors}")

    return schema.load(params)  # type: ignore


def save(crop_params, file="crop_params.csv"):
    """Save crop params in a csv.
    Default path is in OUTPUT_DIR with filename 'crop_params.csv'

    Args:
        file: name or path to csv file

    """
    # Index is 0 if not in the SPP_LIST
    if crop_params.species in SPP_LIST:
        index = SPP_LIST.index(crop_params.species) + 1
    else:
        index = 0

    data = [
        index,
        crop_params.species,
        crop_params.slope,
        crop_params.intercept,
        crop_params.nitrogen_below,
        crop_params.nitrogen_above,
        crop_params.carbon_below,
        crop_params.carbon_above,
        crop_params.root_to_shoot,
    ]
    cols = [
        "Sc",
        "Name",
        "a",
        "b",
        "crop_bgn",
        "crop_agn",
        "crop_bgc",
        "crop_agc",
        "crop_rs",
    ]
    csv_handler.print_csv(file, data, col_names=cols)
