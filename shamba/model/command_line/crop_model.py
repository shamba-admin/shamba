#!/usr/bin/python

"""Module holding Crop class for the crop model."""
from typing import Tuple

import numpy as np
from marshmallow import Schema, fields, post_load

from .. import configuration
from ..common import csv_handler
from .common_schema import OutputSchema as ClimateDataOutputSchema

# from .crop_params import CropParams
from .crop_params import (
    ROOT_IN_TOP_30,
    CropParamsData,
    CropParamsSchema,
)
from .crop_params import from_species_index as create_crop_params_from_species_index

"""
Crop model object. Calculate residues and soil inputs
for given parameters.

Instance variables
------------------
crop_params     CropParams object with crop params (slope, carbon, etc.)
output          output to soil,fire in t C ha^-1
                    (dict with keys 'carbon','nitrogen','DMon','DMoff')

"""


class CropModelData:
    def __init__(self, crop_params, output):
        self.crop_params = crop_params
        self.output = output


class ClimateDataSchema(Schema):
    crop_params = fields.Nested(CropParamsSchema)
    output = fields.Nested(ClimateDataOutputSchema)

    @post_load
    def build(self, data, **kwargs):
        return CropModelData(**data)


def create(crop_params, no_of_years, crop_yield, left_in_field) -> CropModelData:
    """Args:
    crop_params: CropParams object with crop params
    cropYield: dry matter yield of the crop in t C ha^-1
    leftInField: fraction of residues left in field post-harvest

    """
    raw_crop_model_data = {
        "crop_params": vars(
            crop_params
        ),  # We need to convert this to a dictionary for validation
        "output": get_inputs(crop_params, no_of_years, crop_yield, left_in_field),
    }

    schema = ClimateDataSchema()
    errors = schema.validate(raw_crop_model_data)

    if errors != {}:
        print(f"Errors in crop model data: {str(errors)}")

    return schema.load(raw_crop_model_data)


def get_inputs(crop_params, no_of_years, crop_yield, left_in_field):
    """Calculate and return soil carbon inputs, nitrogen inputs,
    on-farm residues, and off-farm residues from soil parameters.

    Use the IPCC slope and intercept values
    outlined in 2006 national GHG assessment guidelines (table 11.2)
    to model

    Args:
        crop_yield: yearly dry-matter crop yield (in t C ha^-1)
        left_in_field: fraction left in field after harvest
    Returns:
        output: dict with soil,fire inputs due to crop
                    (keys='carbon','nitrogen','DMon','DMoff')
    """

    # residues
    res = crop_yield * crop_params.slope + crop_params.intercept
    res *= np.ones(no_of_years)  # convert to array
    resAG = res * left_in_field
    resBG = crop_yield + res
    resBG *= crop_params.root_to_shoot * ROOT_IN_TOP_30

    output = {}

    # Standard outputs - in tonnes of carbon and as vectors
    output["above"] = {
        "carbon": resAG * crop_params.carbon_above,
        "nitrogen": resAG * crop_params.nitrogen_above,
        "DMon": resAG,
        "DMoff": resAG * (1 - left_in_field),
    }
    output["below"] = {
        "carbon": resBG * crop_params.carbon_below,
        "nitrogen": resBG * crop_params.nitrogen_below,
        "DMon": resBG,
        "DMoff": np.zeros(len(res)),
    }

    return output


def save(crop_model, file="crop_model.csv"):
    """Save output of crop model to a csv file.
    Default path is in OUTPUT_DIR.

    Args:
        file: name or path to csv file

    """
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(crop_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)


def get_crop_models_and_crop_params(
    input_data, no_of_years, start_index, end_index, crop_getter
):
    results = [
        crop_getter(input_data, no_of_years, index)
        for index in range(start_index, end_index + 1)
    ]

    # Unzip the results into two separate lists
    crop_models, crop_params = zip(*results)

    # Convert the results to lists (as zip returns tuples)
    return list(crop_models), list(crop_params)


def get_crop_bases(input_data, no_of_years, start_index, end_index):
    return get_crop_models_and_crop_params(
        input_data, no_of_years, start_index, end_index, get_crop_base
    )


def get_crop_projections(input_data, no_of_years, start_index, end_index):
    return get_crop_models_and_crop_params(
        input_data, no_of_years, start_index, end_index, get_crop_projection
    )


def get_crop_data(
    input_data, no_of_years, prefix, index
) -> Tuple[CropModelData, CropParamsData]:
    spp = int(input_data[f"{prefix}_spp{index}"])
    harvYield = np.zeros(no_of_years)
    harvYield[
        int(input_data[f"{prefix}_start{index}"]) : int(
            input_data[f"{prefix}_end{index}"]
        )
    ] = float(input_data[f"{prefix}_yd{index}"])
    harv_frac = float(input_data[f"{prefix}_left{index}"])

    crop_params = create_crop_params_from_species_index(spp)
    crop_model = create(
        crop_params=crop_params,
        no_of_years=no_of_years,
        crop_yield=harvYield,
        left_in_field=harv_frac,
    )

    return crop_model, crop_params


def get_crop_base(
    input_data, no_of_years, index
) -> Tuple[CropModelData, CropParamsData]:
    return get_crop_data(input_data, no_of_years, "crop_base", index)


def get_crop_projection(
    input_data, no_of_years, index
) -> Tuple[CropModelData, CropParamsData]:
    return get_crop_data(input_data, no_of_years, "crop_proj", index)
