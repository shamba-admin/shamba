from typing import Literal
import numpy as np

import model.soil_models.roth_c.inverse_roth_c as roth_c
import model.soil_models.example_soil_model.inverse_example as example_soil_model
from .soil_model_types import SoilModelType, InverseSoilModelData
from ..common import csv_handler


def get_soil_model(soil_model_type: SoilModelType):
    match soil_model_type:
        case SoilModelType.ROTH_C:
            return roth_c
        case SoilModelType.EXAMPLE:
            return example_soil_model
        case _:
            raise ValueError(f"Unknown soil model type: {soil_model_type}")

# NB the format of the following print_to_stdout currently matches requirements for RothC, 
# which is the only soil C model currently available.
def print_to_stdout(inverse_soil_model: InverseSoilModelData):
    pools = ["DPM", "RPM", "BIO", "HUM"]
    print("\nINVERSE CALCULATIONS")
    print("====================\n")
    print("Equilibrium C -", inverse_soil_model.eq_C.sum() + inverse_soil_model.soil.iom)
    for i in range(len(inverse_soil_model.eq_C)):
        print("   ", pools[i], "- - - -", inverse_soil_model.eq_C[i])
    print("    IOM", "- - - -", inverse_soil_model.soil.iom)
    print("Equil. inputs -", inverse_soil_model.input_C)
    print("")


def save(inverse_soil_model: InverseSoilModelData, file="soil_model_inverse.csv"):
    data = np.array(
        [
            np.sum(inverse_soil_model.eq_C) + inverse_soil_model.soil.iom,
            inverse_soil_model.eq_C[0],
            inverse_soil_model.eq_C[1],
            inverse_soil_model.eq_C[2],
            inverse_soil_model.eq_C[3],
            inverse_soil_model.soil.iom,
            inverse_soil_model.input_C,
        ]
    )
    cols = ["Ceq", "dpm", "rpm", "bio", "hum", "iom", "inputs"]
    csv_handler.print_csv(file, data, col_names=cols)
