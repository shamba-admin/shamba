#!/usr/bin/python

"""Module holding Crop class for the crop model."""
import numpy as np
from marshmallow import Schema, fields, post_load

from ..common import csv_handler

from .. import configuration
# from .crop_params import CropParams
from .crop_params import ROOT_IN_TOP_30, CropParamsSchema

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

class ClimateDataOutputFieldSchema(Schema):
    carbon = fields.List(fields.Float, required=True)
    nitrogen = fields.List(fields.Float, required=True)
    DMon = fields.List(fields.Float, required=True)
    DMoff = fields.List(fields.Float, required=True)

class ClimateDataOutputSchema(Schema):
    above = fields.Nested(ClimateDataOutputFieldSchema)
    below = fields.Nested(ClimateDataOutputFieldSchema)
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

def get_crop_model(crop_params, crop_yield, left_in_field):
    """Args:
        crop_params: CropParams object with crop params
        cropYield: dry matter yield of the crop in t C ha^-1
        leftInField: fraction of residues left in field post-harvest

    """
    raw_crop_model_data = {
        "crop_params": vars(crop_params), # We need to convert this to a dictionary for validation
        "output": get_inputs(crop_params, crop_yield, left_in_field)
    }

    schema = ClimateDataSchema()
    errors = schema.validate(raw_crop_model_data)

    print(f"Errors in crop model data: {str(errors)}")

    return schema.load(raw_crop_model_data)

def get_inputs(crop_params, crop_yield, left_in_field):
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
    res *= np.ones(configuration.N_YEARS)  # convert to array
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
