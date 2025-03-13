import os  # Add the parent directory to the Python path
import model.command_line.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
from model.command_line.crop_model import get_crop_bases, get_crop_projects


def test_crop_model():
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    crop_base, _crop_par_base = get_crop_bases(
        input_data=csv_input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )
    crop_project, _crop_par_project = get_crop_projects(
        input_data=csv_input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )

    fire_base = np.zeros(N_YEARS)
    fire_proj = np.zeros(N_YEARS)

    crop_base_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_base, fire=fire_base
    )
    crop_project_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_project, fire=fire_proj
    )

    assert crop_base_emissions == pytest.approx(
        [
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
            0.25450498,
        ]
    )
    assert crop_project_emissions == pytest.approx(
        [
            0.2223257,
            0.2223257,
            0.2223257,
            0.2223257,
            0.2223257,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
            0.09022694,
        ]
    )
