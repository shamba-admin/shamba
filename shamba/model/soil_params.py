from typing import Dict, Any, Tuple
import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler
from model.common.data_sources.soil import get_soil_data


def validate_clay(value):
    return ["Clay value must be between 0 and 100."] if value < 0 or value > 100 else []


def validate_Cy0(value):
    return (
        ["Cy0 value must be between 0 and 10000."] if value < 0 or value > 10000 else []
    )


class SoilParamsData:
    def __init__(self, Cy0, clay, depth, Ceq, iom):
        self.Cy0 = Cy0
        self.clay = clay
        self.depth = depth
        self.Ceq = Ceq
        self.iom = iom


class SoilParamsSchema(Schema):
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

    clay = fields.Float(required=True, validate=lambda v: validate_clay(v))
    Cy0 = fields.Float(required=True, validate=lambda v: validate_Cy0(v))
    depth = fields.Float(required=True)
    Ceq = fields.Float(required=True)
    iom = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return SoilParamsData(**data)


def create(soil_params: Dict[str, Any]) -> SoilParamsData:
    """Create soil data.

    Args:
        soil_params: dict with soil params (keys='Cy0', 'clay')

    Returns:
        SoilParamsData: object containing soil parameters
    """
    Cy0 = soil_params["Cy0"]
    Ceq = 1.25 * Cy0

    params = {
        "Cy0": Cy0,
        "clay": soil_params["clay"],
        "depth": 30.0,
        "Ceq": Ceq,
        "iom": 0.049 * Ceq**1.139,
    }

    schema = SoilParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in soil params: {errors}")

    return schema.load(params)  # type: ignore


def from_csv(filename="soil.csv"):
    """Construct Soil object from csv data."""
    data = csv_handler.read_csv(filename)

    params = {"Cy0": data[0], "clay": data[1]}
    return create(params)


def from_location(location: Tuple[float, float], use_api: bool):
    """Construct Soil object from HWSD data for given location

    Args:
        location: location to look for in HWSD
    Returns:
        Soil object
    """

    result = get_soil_data(location, use_api)

    if result is None:
        raise

    Cy0, clay = result
    params = {"Cy0": Cy0, "clay": clay}
    return create(params)


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
