import os  # Add the parent directory to the Python path
import model.command_line.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.command_line.litter as LitterModel


def test_fertiliser_model():
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    synthetic_fertiliser_base = LitterModel.synthetic_fert(
        freq=int(csv_input_data["base_sf_int"]),
        qty=float(csv_input_data["base_sf_qty"]),
        nitrogen=float(csv_input_data["base_sf_n"]),
        no_of_years=N_YEARS,
    )

    fertiliser_base_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_base]
    )
    fertiliser_projection_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_base]
    )

    assert fertiliser_base_emissions == pytest.approx(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    assert fertiliser_projection_emissions == pytest.approx(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
