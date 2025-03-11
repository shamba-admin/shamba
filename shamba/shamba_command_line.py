#!/usr/bin/python
"""
### TERMS AND CONDITIONS ###
This software is provided under the University of Edinburgh's Open Technology By
downloading this software you accept the University of Edinburgh's Open Technology
terms and conditions.

These can be viewed here: http://www.research-innovation.ed.ac.uk/Opportunities/small-
holder-agriculture-mitigation-benefit-assessment-tool

### INSTRUCTIONS ###
Below are NINE steps to help you run this script.

This script provides a method for estimating GHG impacts for single or multiple
interventions. It has been developed as part of the Plan Vivo approved approach
See here for further details:
https://shambatool.files.wordpress.com/2013/10/shamba-methodology-v9-plan-vivo-approved-approach.pdf

This script runs SHAMBA v1.1 in conjunction with the '_input.csv' sheet
generated from the Excel spreadsheet 'SHAMBA_input_output_template_v1.1'. For
the full instructions, first see this Excel spreadsheet in the
'plan_vivo_approach_excel_templates' folder first .

The script is currently set up as an example to run with the example
'WL_input.csv' generated from the Excel spreadsheet 'example_SHAMBA_input_output_uganda_tech_spec'.
found in the 'plan_vivo_approach_excel_templates' folder.

In the script below, comments with a double hash (##) are
INSTRUCTIONS and require you to do something to the subsequent code.

Comments marked with a single hash (#) are just notation to describe
the subsequent code and shouldn't usually require any action (unless
you want to develop the script to run differently)

If you have feedback on the model code, please feel free to change the code
and send the script with a short explanation to shamba.model@gmail.com
"""

import csv
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from model.common import csv_handler, io_handler

import model.command_line.climate as Climate
import model.command_line.crop_model as CropModel
import model.command_line.crop_params as CropParams
import model.command_line.emit as Emit
import model.command_line.litter as LitterModel
import model.command_line.soil_models.roth_c.forward_roth_c as ForwardRothC
import model.command_line.soil_models.roth_c.inverse_roth_c as InverseRothC
import model.command_line.soil_params as SoilParams
import model.command_line.tree_growth as TreeGrowth
import model.command_line.tree_model as TreeModel
import model.command_line.tree_params as TreeParams
from model import configuration

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_dir))


def get_growth(csv_input_data, spp_key, input_csv, tree_params, allometric_key):
    spp = int(csv_input_data[spp_key])
    if spp == 1:
        growth = TreeGrowth.from_csv1(
            tree_params, n, allometric_key=allometric_key, filename=input_csv
        )
    elif spp == 2:
        growth = TreeGrowth.from_csv2(
            tree_params, n, allometric_key=allometric_key, filename=input_csv
        )
    else:
        growth = TreeGrowth.from_csv3(
            tree_params, n, allometric_key=allometric_key, filename=input_csv
        )

    return growth


def print_crop_emissions(
    crop_base_emissions: np.ndarray,
    crop_project_emissions: np.ndarray,
    crop_difference_emissions: np.ndarray,
):
    table_data = [
        (year, base, proj, proj - base)
        for year, base, proj in zip(
            range(1, len(crop_base_emissions) + 1),
            crop_base_emissions,
            crop_project_emissions,
        )
    ]

    headers = ["Year", "Baseline Emissions", "Projected Emissions", "Difference"]
    table_title = "CROP EMISSIONS (t CO2)"

    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(
        tabulate(
            table_data,
            headers=headers,
            floatfmt=".9f",
            numalign="center",
            tablefmt="fancy_grid",
        )
    )

    print()  # Newline
    print("Total crop difference: ", sum(crop_difference_emissions), " t CO2 ha^-1")
    print("Average crop difference: ", np.mean(crop_difference_emissions))


def print_emissions_table(
    base_emissions, project_emissions, difference, n_years, title
):
    """
    Print a tabular representation of emissions data.

    Args:
        base_emissions (list): List of baseline emissions values.
        project_emissions (list): List of projected emissions values.
        difference (list): List of emission differences.
        n_years (int): Number of years.
        title (str): Title of the emissions table.
    """
    table_data = [
        [
            i + 1,
            f"{base_emissions[i]:.2f}",
            f"{project_emissions[i]:.2f}",
            f"{difference[i]:.2f}",
        ]
        for i in range(n_years)
    ]

    headers = ["Year", "Baseline Emissions", "Projected Emissions", "Difference"]

    print()  # Newline
    print()  # Newline
    print(title)
    print("=" * len(title))
    print(
        tabulate(
            table_data,
            headers=headers,
            floatfmt=".9f",
            numalign="center",
            tablefmt="fancy_grid",
        )
    )
    print()  # Newline
    print(f"Total difference: {sum(difference):.2f}")
    print(f"Average difference: {np.mean(difference):.2f}")


def print_fire_emissions(
    fire_base_emissions, fire_project_emissions, fire_difference, n_years
):
    print_emissions_table(
        fire_base_emissions,
        fire_project_emissions,
        fire_difference,
        n_years,
        "FIRE EMISSIONS (t CO2)",
    )


def print_fertilizer_emissions(
    fertiliser_base_emissions,
    fertiliser_project_emissions,
    fertiliser_difference,
    n_years,
):
    print_emissions_table(
        fertiliser_base_emissions,
        fertiliser_project_emissions,
        fertiliser_difference,
        n_years,
        "FERTILISER EMISSIONS (t CO2)",
    )


def print_litter_emissions(
    litter_base_emissions, litter_project_emissions, litter_difference, n_years
):
    print_emissions_table(
        litter_base_emissions,
        litter_project_emissions,
        litter_difference,
        n_years,
        "LITTER EMISSIONS (t CO2)",
    )


def print_tree_emissions(
    tree_base_emissions, tree_project_emissions, tree_difference, n_years
):
    print_emissions_table(
        tree_base_emissions,
        tree_project_emissions,
        tree_difference,
        n_years,
        "TREE EMISSIONS (t CO2)",
    )


def print_soil_emissions(
    soil_base_emissions, soil_project_emissions, soil_difference, n_years
):
    print_emissions_table(
        soil_base_emissions,
        soil_project_emissions,
        soil_difference,
        n_years,
        "SOIL EMISSIONS (t CO2)",
    )


def print_total_emissions(
    emit_base_emissions, emit_project_emissions, emit_difference, n_years
):
    print_emissions_table(
        emit_base_emissions,
        emit_project_emissions,
        emit_difference,
        n_years,
        "TOTAL EMISSIONS (t CO2)",
    )


def create_tree_params_from_species_index(csv_input_data, tree_count):
    return [
        TreeParams.from_species_index(int(csv_input_data[f"species{i + 1}"]))
        for i in range(tree_count)
    ]


def create_tree_growths(
    csv_input_data, input_csv, tree_params, allometric_key, tree_count
):
    return [
        get_growth(
            csv_input_data,
            f"species{i + 1}",
            input_csv,
            tree_params[i],
            allometric_key=allometric_key,
        )
        for i in range(tree_count)
    ]


def create_tree_projects(
    csv_input_data,
    tree_params,
    growths,
    thinning_project,
    thinning_fraction_left_project,
    mortality_project,
    mortality_fraction_left_project,
    no_of_years,
    tree_count,
):
    return [
        TreeModel.from_defaults(
            tree_params=tree_params[i],
            tree_growth=growths[i],
            yearPlanted=int(csv_input_data[f"proj_plant_yr{i + 1}"]),
            standDens=int(csv_input_data[f"proj_plant_dens{i + 1}"]),
            thin=thinning_project,
            thinFrac=thinning_fraction_left_project,
            mort=mortality_project,
            mortFrac=mortality_fraction_left_project,
            no_of_years=no_of_years,
        )
        for i in range(tree_count)
    ]


def print_tree_projects(tree_projects):
    for project in tree_projects:
        TreeModel.print_biomass(project)
        TreeModel.print_balance(project)


def save_tree_projects(tree_projects, plot_name):
    for i in range(len(tree_projects)):
        TreeModel.save(tree_projects[i], plot_name + f"_tree_proj{i + 1}.csv")


def plot_tree_projects(tree_projects, plot_name):
    for project in tree_projects:
        TreeModel.plot_biomass(project, saveName=plot_name + "_biomassPools.png")
        TreeModel.plot_balance(project, saveName=plot_name + "_massBalance.png")


def setup_project_directory(project_name, arguments):
    """
    Set up a new project directory with the required input files.

    Args:
    project_name (str): The name of the new project directory.

    Returns:
    str: Path to the newly created project directory.
    """

    # New project directory
    project_dir = os.path.join(configuration.PROJECT_DIR, project_name)

    # Input directory within the project directory
    input_dir = os.path.join(project_dir, "input")

    # Create the project and input directories
    os.makedirs(input_dir, exist_ok=True)

    # List of files to copy
    files_to_copy = [
        "crop_ipcc_baseline.csv",
        "crop_ipcc.csv",
        "climate.csv",
        "soil-info.csv",
        arguments["input-file-name"],
    ]

    # Source directory (using an existing project as an example)
    source_dir = os.path.join(
        configuration.PROJECT_DIR, "examples", "UG_TS_2016", "input"
    )

    # Copy each file
    for file in files_to_copy:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(input_dir, file)
        shutil.copy2(source_file, dest_file)
        print(f"Copied {file} to {dest_file}")

    print(f"Project setup complete. New project directory: {project_dir}")
    return project_dir


def main(n, arguments):
    project_name = arguments["project-name"]

    # Create a new project directory
    setup_project_directory(project_name, arguments)

    # Setup the project directory constants
    configuration.SAVE_DIR = os.path.join(configuration.PROJECT_DIR, project_name)

    # specifiying input and output files
    configuration.INPUT_DIR = os.path.join(configuration.SAVE_DIR, "input")
    configuration.OUTPUT_DIR = os.path.join(configuration.SAVE_DIR, "output")

    input_csv = arguments["input-file-name"]

    # ----------
    # getting input data
    # ----------

    ## creating dictionary of input data from input.csv
    file_path = os.path.join(configuration.INPUT_DIR, input_csv)
    csv_input_data = csv_handler.get_csv_input_data(n, file_path)

    # terms in coded below preceded by csv_input_data are values being pulled in from dictionary
    # created above. Converting to float or interger as needed for each
    # key

    ## getting plot anlaysis number to name output

    st = int(csv_input_data["analysis_no"])

    # ----------
    # location information
    # ----------
    loc = (float(csv_input_data["lat"]), float(csv_input_data["lon"]))
    climate = Climate.from_location(loc)

    # ----------
    # project length
    # ----------
    # YEARS = length of tree data. ACCT = years in accounting period
    N_YEARS = int(csv_input_data["yrs_proj"])
    N_ACCT = int(csv_input_data["yrs_acct"])
    N_TREES = 3  # TODO: parameteris this

    # ----------
    # soil equilibrium solve
    # ----------
    soil = SoilParams.from_location(loc)
    invRoth = InverseRothC.create(soil, climate)

    # ----------
    # tree model
    # ----------

    """
    If nitrogen allocations, carbon, root/shoot and/or wood density attributes
    differ between tree cohorts, add a new row specifying these tree parametres
    to the the tree_defaults.csv at shamba/default_input folder and make sure the 
    '_input.csv' file correctly attributes each tree cohort to the relevant 
    parametres under 'trees in baseline' and 'trees in project'
    """

    # linking tree cohort parameteres
    tree_par_base = TreeParams.from_species_index(int(csv_input_data["species_base"]))
    tree_par1 = TreeParams.from_species_index(int(csv_input_data["species1"]))
    tree_par2 = TreeParams.from_species_index(int(csv_input_data["species2"]))
    tree_par3 = TreeParams.from_species_index(int(csv_input_data["species3"]))

    tree_params = create_tree_params_from_species_index(csv_input_data, N_TREES)

    # linking tree growth
    allometric_key = arguments["allometric-key"]

    growth_base = get_growth(
        csv_input_data,
        "species_base",
        input_csv,
        tree_par_base,
        allometric_key=allometric_key,
    )
    growth1 = get_growth(
        csv_input_data, "species1", input_csv, tree_par1, allometric_key=allometric_key
    )
    growth2 = get_growth(
        csv_input_data, "species2", input_csv, tree_par2, allometric_key=allometric_key
    )
    growth3 = get_growth(
        csv_input_data, "species3", input_csv, tree_par3, allometric_key=allometric_key
    )

    tree_growths = create_tree_growths(
        csv_input_data, input_csv, tree_params, allometric_key, N_TREES
    )

    # specify thinning regime and fraction left in field (lif)
    # baseline thinning regime
    # (add line of thinning[yr] = % thinned for each event)
    thinning_base = np.zeros(N_YEARS + 1)
    thinning_base[int(csv_input_data["thin_base_yr1"])] = float(
        csv_input_data["thin_base_pc1"]
    )
    thinning_base[int(csv_input_data["thin_base_yr2"])] = float(
        csv_input_data["thin_base_pc2"]
    )

    # project thinning regime
    # (add need line of thinning[yr] = % thinned for each event)
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

    # baseline fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_base = np.array(
        [
            1,
            float(csv_input_data["thin_base_br"]),
            float(csv_input_data["thin_base_st"]),
            1,
            1,
        ]
    )

    # project fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_project = np.array(
        [
            1,
            float(csv_input_data["thin_proj_br"]),
            float(csv_input_data["thin_proj_st"]),
            1,
            1,
        ]
    )

    # specify mortality regime and fraction left in field (lif)

    # baseline yearly mortality
    mortality_base = np.array((N_YEARS + 1) * [float(csv_input_data["base_mort"])])

    # project yearly mortality
    mortality_project = np.array((N_YEARS + 1) * [float(csv_input_data["proj_mort"])])

    # baseline fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_base = np.array(
        [
            1,
            float(csv_input_data["mort_base_br"]),
            float(csv_input_data["mort_base_st"]),
            1,
            1,
        ]
    )

    # project fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_project = np.array(
        [
            1,
            float(csv_input_data["mort_proj_br"]),
            float(csv_input_data["mort_proj_st"]),
            1,
            1,
        ]
    )

    # run tree model

    # trees planted in baseline (standDens must be at least 1)
    tree_base = TreeModel.from_defaults(
        tree_params=tree_par1,
        tree_growth=growth_base,
        yearPlanted=0,
        standDens=int(csv_input_data["base_plant_dens"]),
        thin=thinning_base,
        thinFrac=thinning_fraction_left_base,
        mort=mortality_base,
        mortFrac=mortality_fraction_left_base,
        no_of_years=N_YEARS,
    )

    tree_projects = create_tree_projects(
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

    # ----------
    # Fire model
    # ----------
    # return interval of fire, [::2] = 1 is return interval of two years
    base_fire_interval = int(csv_input_data["fire_int_base"])
    if base_fire_interval == 0:
        fire_base = np.zeros(N_YEARS)
    else:
        fire_base = np.zeros(N_YEARS)
        fire_base[::base_fire_interval] = int(csv_input_data["fire_pres_base"])

    project_fire_interval = int(csv_input_data["fire_int_proj"])
    if project_fire_interval == 0:
        fire_project = np.zeros(N_YEARS)
    else:
        fire_project = np.zeros(N_YEARS)
        fire_project[::project_fire_interval] = int(csv_input_data["fire_pres_proj"])

    # ----------
    # Litter model
    # ----------
    # baseline external organic inputs
    litter_external_base = LitterModel.from_defaults(
        litterFreq=int(csv_input_data["base_lit_int"]),
        litterQty=float(csv_input_data["base_lit_qty"]),
        no_of_years=N_YEARS,
    )

    # baseline synthetic fertiliser additions
    synthetic_fertiliser_base = LitterModel.synthetic_fert(
        freq=int(csv_input_data["base_sf_int"]),
        qty=float(csv_input_data["base_sf_qty"]),
        nitrogen=float(csv_input_data["base_sf_n"]),
        no_of_years=N_YEARS,
    )

    # Project external organic inputs
    litter_external_project = LitterModel.from_defaults(
        litterFreq=int(csv_input_data["proj_lit_int"]),
        litterQty=float(csv_input_data["proj_lit_qty"]),
        no_of_years=N_YEARS,
    )

    # Project synthetic fertiliser additions
    synthetic_fertiliser_project = LitterModel.synthetic_fert(
        freq=int(csv_input_data["proj_sf_int"]),
        qty=float(csv_input_data["proj_sf_qty"]),
        nitrogen=float(csv_input_data["proj_sf_n"]),
        no_of_years=N_YEARS,
    )

    # ----------
    # Crop model
    # ----------
    # Baseline specify crop, yield, and % left in field in csv file
    cropPar = csv_handler.read_csv(input_csv)
    cropPar = np.atleast_2d(cropPar)

    crop_base, crop_par_base = CropModel.get_crop_bases(
        input_data=csv_input_data,
        no_of_years=N_YEARS,
        start_index=1,
        end_index=3,
    )
    crop_project, crop_par_project = CropModel.get_crop_projects(
        input_data=csv_input_data,
        no_of_years=N_YEARS,
        start_index=1,
        end_index=3,
    )

    # soil cover for baseline
    cover_base = np.zeros(12)
    cover_base[
        int(csv_input_data["base_cvr_mth_st"]) : int(csv_input_data["base_cvr_mth_en"])
    ] = int(csv_input_data["base_cvr_pres"])

    # soil cover for project
    cover_proj = np.zeros(12)
    cover_proj[
        int(csv_input_data["proj_cvr_mth_st"]) : int(csv_input_data["proj_cvr_mth_en"])
    ] = int(csv_input_data["proj_cvr_pres"])

    # Solve to y=0
    forRoth = ForwardRothC.create(
        soil,
        climate,
        cover_base,
        no_of_years=N_YEARS,
        Ci=invRoth.eqC,
        crop=crop_base,
        fire=fire_base,
        solveToValue=True,
    )

    # Soil carbon for baseline and project
    roth_base = ForwardRothC.create(
        soil=soil,
        climate=climate,
        cover=cover_base,
        Ci=forRoth.SOC[-1],
        no_of_years=N_YEARS,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fire=fire_base,
    )

    roth_proj = ForwardRothC.create(
        soil,
        climate,
        cover_proj,
        Ci=forRoth.SOC[-1],
        no_of_years=N_YEARS,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fire=fire_project,
    )

    # Emissions stuff
    emit_base_emissions = Emit.create(
        no_of_years=N_YEARS,
        forRothC=roth_base,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fert=[synthetic_fertiliser_base],
        fire=fire_base,
    )
    emit_project_emissions = Emit.create(
        no_of_years=N_YEARS,
        forRothC=roth_proj,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fert=[synthetic_fertiliser_project],
        fire=fire_project,
    )

    # ----------
    # Printing outputs
    # ----------

    # Print some stuff?
    print("location: ", loc)
    Climate.print_to_stdout(climate)
    SoilParams.print_to_stdout(soil)
    TreeGrowth.print_to_stdout(growth1, label="growth1")
    TreeGrowth.print_to_stdout(growth2, label="growth2")
    TreeGrowth.print_to_stdout(growth3, label="growth3")

    print_tree_projects(tree_projects)

    ForwardRothC.print_to_stdout(forRoth, no_of_years=N_YEARS, label="initialisation")
    ForwardRothC.print_to_stdout(roth_base, no_of_years=N_YEARS, label="baseline")
    ForwardRothC.print_to_stdout(roth_proj, no_of_years=N_YEARS, label="project")
    # =============================================================================

    # Crop Emissions
    crop_base_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_base, fire=fire_base
    )
    crop_project_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_project, fire=fire_project
    )
    crop_difference = crop_project_emissions - crop_base_emissions

    print_crop_emissions(crop_base_emissions, crop_project_emissions, crop_difference)
    # =============================================================================

    # Fertilizer Emissions
    fertiliser_base_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_base]
    )
    fertiliser_project_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_project]
    )
    fertiliser_difference = fertiliser_project_emissions - fertiliser_base_emissions

    print_fertilizer_emissions(
        fertiliser_base_emissions=fertiliser_base_emissions,
        fertiliser_project_emissions=fertiliser_project_emissions,
        fertiliser_difference=fertiliser_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Litter Emissions
    litter_base_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_base], fire=fire_base
    )
    litter_project_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_project], fire=fire_project
    )
    litter_difference = litter_project_emissions - litter_base_emissions

    print_litter_emissions(
        litter_base_emissions=litter_base_emissions,
        litter_project_emissions=litter_project_emissions,
        litter_difference=litter_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Fire Emissions
    fire_base_emissions = Emit.create(no_of_years=N_YEARS, fire=fire_base)
    fire_project_emissions = Emit.create(no_of_years=N_YEARS, fire=fire_project)
    fire_difference = fire_project_emissions - fire_base_emissions

    print_fire_emissions(
        fire_base_emissions=fire_base_emissions,
        fire_project_emissions=fire_project_emissions,
        fire_difference=fire_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Tree Eemissions
    tree_base_emissions = Emit.create(
        no_of_years=N_YEARS, tree=[tree_base], fire=fire_base
    )
    tree_project_emissions = Emit.create(
        no_of_years=N_YEARS,
        tree=tree_projects,
        fire=fire_project,
    )
    tree_difference = tree_project_emissions - tree_base_emissions

    print_tree_emissions(
        tree_base_emissions=tree_base_emissions,
        tree_project_emissions=tree_project_emissions,
        tree_difference=tree_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Soil Emissions
    soil_base_emissions = emit_base_emissions - (
        crop_base_emissions
        + fertiliser_base_emissions
        + litter_base_emissions
        + fire_base_emissions
        + tree_base_emissions
    )
    soil_project_emissions = emit_project_emissions - (
        crop_project_emissions
        + fertiliser_project_emissions
        + litter_project_emissions
        + fire_project_emissions
        + tree_project_emissions
    )
    soil_difference = soil_project_emissions - soil_base_emissions

    print_soil_emissions(
        soil_base_emissions=soil_base_emissions,
        soil_project_emissions=soil_project_emissions,
        soil_difference=soil_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Total Emissions
    emit_difference = emit_project_emissions - emit_base_emissions

    print_total_emissions(
        emit_base_emissions=emit_base_emissions,
        emit_project_emissions=emit_project_emissions,
        emit_difference=emit_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Summary of GHG pools
    summary_difference_data = [
        ["Difference Type", "Value", "Units"],
        ["Total Crop Difference", f"{sum(crop_difference):.2f}", "t CO2 ha^-1"],
        [
            "Total Fertiliser Difference",
            f"{sum(fertiliser_difference):.2f}",
            "t CO2 ha^-1",
        ],
        ["Total Litter Difference", f"{sum(litter_difference):.2f}", "t CO2 ha^-1"],
        ["Total Fire Difference", f"{sum(fire_difference):.2f}", "t CO2 ha^-1"],
        ["Total Tree Difference", f"{sum(tree_difference):.2f}", "t CO2 ha^-1"],
        ["Total Soil Difference", f"{sum(soil_difference):.2f}", "t CO2 ha^-1"],
        ["Total Difference", f"{sum(emit_difference):.2f}", "t CO2 ha^-1"],
    ]

    accounting_year = csv_input_data["yrs_acct"]

    summary_difference_title = (
        f"SUMMARY OF EMISSIONS for Year {accounting_year} (t CO2)"
    )

    print()  # Newline
    print()  # Newline
    print(summary_difference_title)
    print("=" * len(summary_difference_title))
    print(tabulate(summary_difference_data, tablefmt="fancy_grid"))
    # =============================================================================

    # Save stuff

    # starting plot output number
    st = 1

    dir = configuration.OUTPUT_DIR + "_" + mod_run + "\plot_" + str(n + st)

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    plot_name = dir + "\plot_" + str(n + st)

    Climate.save(climate, plot_name + "_climate.csv")

    SoilParams.save(soil, plot_name + "_soil.csv")
    TreeGrowth.save(growth1, plot_name + "_growth1.csv")
    TreeGrowth.save(growth2, plot_name + "_growth2.csv")
    TreeGrowth.save(growth3, plot_name + "_growth3.csv")

    save_tree_projects(tree_projects, plot_name=plot_name)

    i = 1
    for i in range(len(crop_base)):
        CropModel.save(crop_base[i], plot_name + "_crop_model_base_" + str(i) + ".csv")

        CropParams.save(
            crop_par_base[i], plot_name + "_crop_params_base_" + str(i) + ".csv"
        )

        CropModel.save(
            crop_project[i], plot_name + "_crop_model_proj_" + str(i) + ".csv"
        )

        CropParams.save(
            crop_par_project[i], plot_name + "_crop_params_proj_" + str(i) + ".csv"
        )

    InverseRothC.save(invRoth, plot_name + "_invRoth.csv")
    ForwardRothC.save(
        forward_roth_c=forRoth, no_of_years=N_YEARS, file=plot_name + "_forRoth.csv"
    )

    ForwardRothC.save(
        forward_roth_c=roth_base,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_base.csv",
    )
    ForwardRothC.save(
        forward_roth_c=roth_proj,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_proj.csv",
    )

    Emit.save(emit_base_emissions, emit_project_emissions, plot_name + "_emit_proj.csv")

    with open(
        configuration.OUTPUT_DIR
        + "_"
        + mod_run
        + "\plot_"
        + str(n + st)
        + "\plot_"
        + str(n + st)
        + "_emissions_all_pools_per_year.csv",
        "w+",
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [
                "emit_base_emissions",
                "emit_project_emissions",
                "emit_difference",
                "soil_base",
                "soil_proj",
                "soil_difference",
                "tree_base",
                "tree_proj",
                "tree_difference",
                "fire_base",
                "fire_project",
                "fire_difference",
                "lit_base",
                "lit_proj",
                "litter_difference",
                "fert_base",
                "fert_proj",
                "fertiliser_difference",
                "crop_base",
                "crop_project",
                "crop_difference",
            ]
        )
        for i in range(len(emit_base_emissions)):
            writer.writerow(
                [
                    emit_base_emissions[i],
                    emit_project_emissions[i],
                    emit_difference[i],
                    soil_base_emissions[i],
                    soil_project_emissions[i],
                    soil_difference[i],
                    tree_base_emissions[i],
                    tree_project_emissions[i],
                    tree_difference[i],
                    fire_base_emissions[i],
                    fire_project_emissions[i],
                    fire_difference[i],
                    litter_base_emissions[i],
                    litter_project_emissions[i],
                    litter_difference[i],
                    fertiliser_base_emissions[i],
                    fertiliser_project_emissions[i],
                    fertiliser_difference[i],
                    crop_base_emissions[i],
                    crop_project_emissions[i],
                    crop_difference[i],
                ]
            )

    # Plot stuff
    plot_tree_projects(tree_projects, plot_name)

    plt.close()

    ForwardRothC.plot(forRoth, no_of_years=N_YEARS, legendStr="initialisation")

    ForwardRothC.plot(roth_base, no_of_years=N_YEARS, legendStr="baseline")

    ForwardRothC.plot(
        roth_proj,
        no_of_years=N_YEARS,
        legendStr="project",
        saveName=plot_name + "_soilModel.png",
    )
    plt.close()

    Emit.plot(emit_base_emissions, legendStr="baseline")
    Emit.plot(emit_project_emissions, legendStr="project")

    # TODO: what is this? How could `ax` be attached to Emission?
    # Why is it done here instead of in the plot function?
    # Commenting out for now
    # emit.Emission.ax.plot(emit_difference, label="difference")
    # emit.Emission.ax.legend(loc="best")

    plt.savefig(os.path.join(configuration.OUTPUT_DIR, plot_name + "_emissions.png"))
    plt.close()

    Emit.save(
        emit_base_emissions=emit_base_emissions,
        emit_proj_emissions=emit_project_emissions,
        file=plot_name + "_emissions.csv",
    )

    return (
        sum(crop_difference),
        sum(fertiliser_difference),
        sum(litter_difference),
        sum(fire_difference),
        sum(tree_difference),
        sum(soil_difference),
        sum(emit_difference),
    )


if __name__ == "__main__":
    """
    ## STEP ## (OPTIONAL)
    If you only want to run a selection of rows (i.e. scenarios) from the '_input.csv'
    file, specify the number_of_rows value here. If you want to run all rows place a # in front
    of the line of code below to deactivate them.
    """

    number_of_rows = 1
    # Get command line arguments
    arguments = io_handler.get_arguments_interactively()

    """
    ## STEP 8 ##
    Specify in the code below the title that will be attached to all of your
    output folders and files
    """
    mod_run = arguments["output-title"]

    emit_output_data = []
    for n in range(number_of_rows):
        emit_output_data.append(main(n, arguments))

    """
    ## STEP 9 ##
    After you have completed all of the above steps, run the code.
    
    Results for each line of the _input.csv file will (i.e. each model run) 
    will appear in a subfolder in the shamba/projects/"project"/output_/. 
    
    All results are in tCO2e.
    
    The .csv output files provided detailed information about the parameteres 
    (params), baseline emissions (base) and intervention emissions (proj) for
    each GHG sink or source. The .csv summarise '_emissions_all_pools_per_year.csv' this information.
    You can analyse this information further using Excel.
    """
