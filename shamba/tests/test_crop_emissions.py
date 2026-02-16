import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
from model.crop_model import get_crop_bases, get_crop_projects


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
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
            0.22412858,
        ]
    )
    assert crop_project_emissions == pytest.approx(
        [
            0.19579005,
            0.19579005,
            0.19579005,
            0.19579005,
            0.19579005,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
            0.07945792,
        ]
    )
