import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
from model.crop_model import get_crop_bases, get_crop_projects
import model.common.constants as CONSTANTS


def test_fire_model():
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    crop_base, _crop_par_base = get_crop_bases(
        input_data=csv_input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )
    crop_project, _crop_par_project = get_crop_projects(
        input_data=csv_input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )

    base_fire_interval = int(2)
    if base_fire_interval == 0:
        fire_base = np.zeros(N_YEARS)
    else:
        fire_base = np.zeros(N_YEARS)
        fire_base[::base_fire_interval] = int(1)

    proj_fire_interval = int(csv_input_data["fire_int_proj"])
    if proj_fire_interval == 0:
        fire_project = np.zeros(N_YEARS)
    else:
        fire_project = np.zeros(N_YEARS)
        fire_project[::proj_fire_interval] = int(csv_input_data["fire_pres_proj"])

    fire_base_emissions = Emit.fire_emit(
        no_of_years=N_YEARS,
        fire=fire_base,
        crop=crop_base,
        tree=[],
        litter=[],
        burn_off=False,
        gwp=CONSTANTS.GWP_AR6,
    )
    fire_project_emissions = Emit.fire_emit(
        no_of_years=N_YEARS,
        fire=fire_project,
        crop=crop_project,
        tree=[],
        litter=[],
        burn_off=False,
        gwp=CONSTANTS.GWP_AR6,
    )

    assert fire_base_emissions == pytest.approx(
        [
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
            0.49552906,
            0,
        ]
    )
    assert fire_project_emissions == pytest.approx(
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
