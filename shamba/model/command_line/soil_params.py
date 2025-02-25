#!/usr/bin/python

"""Module containing Soil class."""

import logging as log
import os
import sys

import numpy as np
from osgeo import gdal
from marshmallow import Schema, fields, post_load

from model.common import csv_handler
from rasters import soil as soil_raster

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


def validate_clay(value):
    return ["Clay value must be between 0 and 100."] if value < 0 or value > 100 else []


def validate_Cy0(value):
    return (
        ["Cy0 value must be between 0 and 10000."] if value < 0 or value > 10000 else []
    )


class SoilParamsData:
    def __init__(self, Cy0, clay):
        self.Cy0 = Cy0
        self.clay = clay
        self.depth = 30.0
        self.Ceq = 1.25 * self.Cy0
        self.iom = 0.049 * self.Ceq**1.139


class SoilParamsSchema(Schema):
    clay = fields.Float(required=True, validate=lambda v: validate_clay(v))
    Cy0 = fields.Float(required=True, validate=lambda v: validate_Cy0(v))

    @post_load
    def build(self, data, **kwargs):
        return SoilParamsData(**data)


def create(soil_params):
    """Initialise soil data.

    Args:
        soil_params: dict with soil params (keys='Cy0,'clay')
    Raises:
        KeyError: if dict doesn't have the right keys
        TypeError: if non-numeric type is used for soil_params

    """
    params = {"Cy0": soil_params["Cy0"], "clay": soil_params["clay"]}

    schema = SoilParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in soil params: {errors}")

    return schema.load(params)


def from_csv(filename="soil.csv"):
    """Construct Soil object from csv data."""
    data = csv_handler.read_csv(filename)

    params = {"Cy0": data[0], "clay": data[1]}
    return create(params)


def from_location(location):
    """Construct Soil object from HWSD data for given location

    Args:
        location: location to look for in HWSD
    Returns:
        Soil object

    """

    mu_global = get_identifier(location)
    result = get_data_from_identifier(mu_global)

    if result is None:
        log.error("COULD NOT FIND %d IN HWSD_DATA.csv", mu_global)
        raise

    Cy0, clay = result
    params = {"Cy0": Cy0, "clay": clay}
    return create(params)


def sanitize_params(params):
    """Check that provided soil data makes sense.

    Args:
        params: soil params dict with keys 'Cy0' and 'clay'
    """

    Cy0 = params["Cy0"]
    clay = params["clay"]

    if clay < 0 or clay > 100 or Cy0 < 0 or Cy0 > 10000:
        log.warning("Unusual soil parameters. Please check.")


def print_to_stdout(soil_params):
    """Print soil information to stdout."""
    print("\nSOIL INFORMATION")
    print("================\n")
    print("Equilibrium C -", soil_params.Ceq)
    print("C at y=0  - - -", soil_params.Cy0)
    print("IOM - - - - - -", soil_params.iom)
    print("Clay  - - - - -", soil_params.clay)
    print("Depth - - - - -", soil_params.depth)
    print("")


def save(soil_params, file="soil_params.csv"):
    """Save soil params to a csv file. Default path is in OUTPUT_DIR
    with filename soil_params.csv

    Args:
        file: name or path to csv file. If path is not given
                (only name), put in OUTPUT_DIR for this program run.

    """
    data = np.array(
        [
            soil_params.Cy0,
            soil_params.clay,
            soil_params.Ceq,
            soil_params.iom,
            soil_params.depth,
        ]
    )
    cols = ["Cy0", "clay", "Ceq", "iom", "depth"]
    csv_handler.print_csv(file, data, col_names=cols)


def get_identifier(location):
    """Find MU_GLOBAL for given location from the HWSD .bil raster."""

    y = location[0]  # lat
    x = location[1]  # long

    # gdal setup
    gdal.AllRegister()
    driver = gdal.GetDriverByName("HFA")
    driver.Register()
    gdal.UseExceptions()

    # open file
    try:
        filename = os.path.join(
            os.path.dirname(os.path.abspath(soil_raster.__file__)), "hwsd.bil"
        )
        ds = gdal.Open(filename)
    except RuntimeError:
        raise csv_handler.FileOpenError("hwsd.bil")

    # TODO: check if these are needed
    # cols = ds.RasterXSize
    # rows = ds.RasterYSize
    # bands = ds.RasterCount

    # georeference info
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    width = transform[1]  # x resolution
    height = transform[5]  # y resolution

    # FIND VALUES
    # cast as ints
    xInt = int((x - xOrigin) / width)
    yInt = int((y - yOrigin) / height)
    band = ds.GetRasterBand(1)  # one-indexed

    data = band.ReadAsArray(xInt, yInt, 1, 1)
    value = data[0, 0]  # MU_GLOBAL for input to HWSD_data.csv

    return value


def get_data_from_identifier(mu):
    """Get soil data from csv given MU_GLOBAL from the raster."""

    filename = os.path.join(
        os.path.dirname(os.path.abspath(soil_raster.__file__)), "HWSD_data.csv"
    )

    soilTable = csv_handler.read_mixed_csv(
        filename,
        cols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        types=(
            int,
            int,
            float,
            int,
            "|S25",
            float,
            float,
            float,
            "|S15",
            float,
            float,
            float,
            float,
        ),
    )

    # Find rows with mu in the MU_GLOBAL column (column 1)
    muRows = []  # rows we want as list of records
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
    cy0 /= 100  # account for percentages
    clay /= 100

    return cy0, clay


# TODO: confirm this isn't being used. The Soil model exists in `project_model.py` and is used in the GUI.
# This module should not depend on that module I think?
# if __name__ == "__main__":
#     data = csv_handler.read_csv(
#         os.path.join(
#             configuration.SHAMBA_DIR, "rasters", "soil", "muGlobalTestValues.csv"
#         )
#     )
#     long = data[:, 1]
#     lat = data[:, 2]
#     mu = data[:, 3]

#     for i in range(len(mu)):
#         a = Soil((lat[i], int[i]))
#         print("\nlocation = %f, %f " % (lat[i], int[i]))
#         print("mu actual= %d" % mu[i])
#         print("mu python= %d" % a.mu_global)
