from typing import Literal

import model.soil_models.roth_c.inverse_roth_c as roth_c
import model.soil_models.example_soil_model.inverse_example as example_soil_model
from .soil_model_type import SoilModelType


def get_soil_model(soil_model_type: SoilModelType):
    match soil_type:
        case SoilModelType.ROTH_C:
            return roth_c
        case SoilModelType.EXAMPLE:
            return example_soil_model
        case _:
            raise ValueError(f"Unknown soil model type: {soil_model_type}")


def print_to_stdout(inverse_roth_c):
    """Print data from inverse RothC run to stdout."""

    pools = ["DPM", "RPM", "BIO", "HUM"]
    print("\nINVERSE CALCULATIONS")
    print("====================\n")
    print("Equilibrium C -", inverse_roth_c.eq_C.sum() + inverse_roth_c.soil.iom)
    for i in range(len(inverse_roth_c.eq_C)):
        print("   ", pools[i], "- - - -", inverse_roth_c.eq_C[i])
    print("    IOM", "- - - -", inverse_roth_c.soil.iom)
    print("Equil. inputs -", inverse_roth_c.input_C)
    print("")


def save(inverse_roth_c, file="soil_model_inverse.csv"):
    """Save data to csv. Default path is OUTPUT_DIR."""

    data = np.array(
        [
            np.sum(inverse_roth_c.eq_C) + inverse_roth_c.soil.iom,
            inverse_roth_c.eq_C[0],
            inverse_roth_c.eq_C[1],
            inverse_roth_c.eq_C[2],
            inverse_roth_c.eq_C[3],
            inverse_roth_c.soil.iom,
            inverse_roth_c.input_C,
        ]
    )
    cols = ["Ceq", "dpm", "rpm", "bio", "hum", "iom", "inputs"]
    csv_handler.print_csv(file, data, col_names=cols)
