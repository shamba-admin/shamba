import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.litter as LitterModel


def test_crop_model():
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    base_fire_interval = int(csv_input_data["fire_int_base"])
    if base_fire_interval == 0:
        fire_base = np.zeros(N_YEARS)
    else:
        fire_base = np.zeros(N_YEARS)
        fire_base[::base_fire_interval] = int(csv_input_data["fire_pres_base"])

    proj_fire_interval = int(csv_input_data["fire_int_proj"])
    if proj_fire_interval == 0:
        fire_project = np.zeros(N_YEARS)
    else:
        fire_project = np.zeros(N_YEARS)
        fire_project[::proj_fire_interval] = int(csv_input_data["fire_pres_proj"])

    litter_external_base = LitterModel.from_defaults(
        litter_frequency=int(csv_input_data["base_lit_int"]),
        litter_quantity=float(csv_input_data["base_lit_qty"]),
        no_of_years=N_YEARS,
    )
    litter_external_project = LitterModel.from_defaults(
        litter_frequency=int(csv_input_data["proj_lit_int"]),
        litter_quantity=float(csv_input_data["proj_lit_qty"]),
        no_of_years=N_YEARS,
    )

    litter_base_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_base], fire=fire_base
    )
    litter_project_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_project], fire=fire_project
    )

    assert litter_base_emissions == pytest.approx(
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
    assert litter_project_emissions == pytest.approx(
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
