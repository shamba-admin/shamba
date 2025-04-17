import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.tree_model as TreeModel
import model.tree_params as TreeParams
import model.tree_growth as TreeGrowth


def test_crop_model():
    input_csv = "WL_input.csv"
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", input_csv)
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])
    N_TREES = 3

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

    tree_par_base = TreeParams.from_species_index(int(csv_input_data["species_base"]))
    tree_params_1 = TreeParams.from_species_index(int(csv_input_data["species1"]))

    thinning_base = np.zeros(N_YEARS + 1)
    thinning_base[int(csv_input_data["thin_base_yr1"])] = float(
        csv_input_data["thin_base_pc1"]
    )
    thinning_base[int(csv_input_data["thin_base_yr2"])] = float(
        csv_input_data["thin_base_pc2"]
    )

    growth_base = TreeGrowth.get_growth(
        csv_input_data,
        "species_base",
        tree_par_base,
        allometric_key="chave dry",
    )

    thinning_fraction_left_base = np.array(
        [
            1,
            float(csv_input_data["thin_base_br"]),
            float(csv_input_data["thin_base_st"]),
            1,
            1,
        ]
    )

    mortality_base = np.array((N_YEARS + 1) * [float(csv_input_data["base_mort"])])

    mortality_fraction_left_base = np.array(
        [
            1,
            float(csv_input_data["mort_base_br"]),
            float(csv_input_data["mort_base_st"]),
            1,
            1,
        ]
    )

    tree_base = TreeModel.from_defaults(
        tree_params=tree_params_1,
        tree_growth=growth_base,
        yearPlanted=0,
        standard_density=int(csv_input_data["base_plant_dens"]),
        thin=thinning_base,
        thinFrac=thinning_fraction_left_base,
        mort=mortality_base,
        mortFrac=mortality_fraction_left_base,
        no_of_years=N_YEARS,
    )

    tree_params = TreeParams.create_tree_params_from_species_index(
        csv_input_data, N_TREES
    )
    tree_growths = TreeGrowth.create_tree_growths(
        csv_input_data, tree_params, "chave dry", N_TREES
    )
    thinning_project = np.zeros(N_YEARS + 1)
    thinning_project[int(csv_input_data["thin_proj_yr1"])] = float(
        csv_input_data["thin_proj_pc1"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr2"])] = float(
        csv_input_data["thin_proj_pc2"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr3"])] = float(
        csv_input_data["thin_proj_pc3"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr4"])] = float(
        csv_input_data["thin_proj_pc4"]
    )
    thinning_fraction_left_project = np.array(
        [
            1,
            float(csv_input_data["thin_proj_br"]),
            float(csv_input_data["thin_proj_st"]),
            1,
            1,
        ]
    )
    mortality_project = np.array((N_YEARS + 1) * [float(csv_input_data["proj_mort"])])
    mortality_fraction_left_project = np.array(
        [
            1,
            float(csv_input_data["mort_proj_br"]),
            float(csv_input_data["mort_proj_st"]),
            1,
            1,
        ]
    )

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=csv_input_data,
        tree_params=tree_params,
        growths=tree_growths,
        thinning_project=thinning_project,
        thinning_fraction_left_project=thinning_fraction_left_project,
        mortality_project=mortality_project,
        mortality_fraction_left_project=mortality_fraction_left_project,
        no_of_years=N_YEARS,
        tree_count=N_TREES,
    )

    tree_base_emissions = Emit.create(
        no_of_years=N_YEARS, tree=[tree_base], fire=fire_base
    )
    tree_project_emissions = Emit.create(
        no_of_years=N_YEARS,
        tree=tree_projects,
        fire=fire_project,
    )

    assert tree_base_emissions == pytest.approx(
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
    assert tree_project_emissions == pytest.approx(
        [
            10.313230851931625,
            -5.681632974584525,
            -8.115516605140044,
            -8.806522263500227,
            -9.33467362728289,
            -9.844150775646117,
            -3.067715123894557,
            -10.46324909645775,
            -11.005403550317887,
            -3.2466032209163287,
            -11.605116087839727,
            -12.156180788516766,
            -12.692044042395144,
            -13.22970366689084,
            4.514811889994492,
            -13.289228704686316,
            -13.81678452873205,
            -14.288309437233384,
            -14.742155478863882,
            -15.173383114802688,
            -15.576724503056745,
            -15.946646055908701,
            -16.277416770410376,
            -16.56319172328749,
            -16.798111226971013,
            -16.97641489216745,
            -17.092568846371886,
            -17.141403367313682,
            -17.11825718724876,
            -17.0191237661273,
            -16.840793999944207,
            -16.580989214605463,
            -16.2384779834415,
            -15.813170373911682,
            205.06420139476623,
            -8.266727963675166,
            -8.28577661540507,
            -7.812857607596557,
            -7.30243529543792,
            -6.759506078315824,
            -6.18910778803017,
            -5.596566893103336,
            -4.987406103300538,
            -4.367224132163187,
            -3.741572745873035,
            -3.115840270359118,
            -2.49514654033794,
            -1.8842527683058052,
            -1.2874886771940632,
            -0.7086981514400357,
        ]
    )
