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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from model.common import csv_handler, io_handler

import model.climate as Climate
import model.crop_model as CropModel
import model.crop_params as CropParams
import model.emit as Emit
import model.litter as LitterModel
import model.soil_models.roth_c.forward_roth_c as ForwardRothC
import model.soil_models.roth_c.inverse_roth_c as InverseRothC
import model.soil_params as SoilParams
import model.tree_growth as TreeGrowth
import model.tree_model as TreeModel
import model.tree_params as TreeParams
from model import configuration
from model.common.calculate_emissions import handle_intervention

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_dir))


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


def print_tree_projects(tree_projects):
    for project in tree_projects:
        TreeModel.print_biomass(project)
        TreeModel.print_balance(project)


def save_tree_projects(tree_projects, plot_name):
    for i in range(len(tree_projects)):
        TreeModel.save(tree_projects[i], plot_name + f"_tree_proj{i + 1}.csv")


def plot_tree_projects(tree_projects, plot_name):
    for project in tree_projects:
        TreeModel.plot_biomass(project, save_name=plot_name + "_biomassPools.png")
        TreeModel.plot_balance(project, save_name=plot_name + "_massBalance.png")


def print_tree_growths(tree_growths):
    for i in range(len(tree_growths)):
        TreeGrowth.print_to_stdout(tree_growths[i], label=f"growth{i+1}")


def save_tree_growths(tree_growths, plot_name):
    for i in range(len(tree_growths)):
        TreeGrowth.save(tree_growths[i], plot_name + f"_growth{i+1}.csv")


def save_crop_data(base_data, project_data, plot_name, model_type):
    for i, (base, project) in enumerate(zip(base_data, project_data), 1):
        base_filename = f"{plot_name}_{model_type}_base_{i}.csv"
        project_filename = f"{plot_name}_{model_type}_proj_{i}.csv"

        if model_type == "crop_model":
            CropModel.save(base, str(base_filename))
            CropModel.save(project, str(project_filename))
        elif model_type == "crop_params":
            CropParams.save(base, str(base_filename))
            CropParams.save(project, str(project_filename))


def write_emissions_csv(configuration, mod_run, n, st, data):
    # Define the output directory and file name
    output_dir = Path(configuration.OUTPUT_DIR + f"_{mod_run}/plot_{n+st}")
    output_file = output_dir / f"plot_{n+st}_emissions_all_pools_per_year.csv"

    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the header and data rows
    header = [
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

    # Write the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(zip(*data.values()))


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
    # project length
    # ----------
    # YEARS = length of tree data. ACCT = years in accounting period
    N_YEARS = int(csv_input_data["yrs_proj"])
    N_TREES = 3  # TODO: parameteris this

    allometric_key = arguments["allometric-key"]

    intervention_emissions = handle_intervention(
        intervention_input=csv_input_data,
        allometry=allometric_key,
        no_of_trees=N_TREES,
        use_api=arguments["use-api"],
    )

    # ----------
    # Printing outputs
    # ----------

    # Print some stuff?
    Climate.print_to_stdout(intervention_emissions.climate)
    SoilParams.print_to_stdout(intervention_emissions.soil)

    print_tree_growths(intervention_emissions.tree_growths)

    print_tree_projects(intervention_emissions.tree_projects)

    ForwardRothC.print_to_stdout(
        intervention_emissions.for_roth, no_of_years=N_YEARS, label="initialisation"
    )
    ForwardRothC.print_to_stdout(
        intervention_emissions.roth_base, no_of_years=N_YEARS, label="baseline"
    )
    ForwardRothC.print_to_stdout(
        intervention_emissions.roth_project, no_of_years=N_YEARS, label="project"
    )
    # =============================================================================

    # Crop Emissions
    print_crop_emissions(
        intervention_emissions.crop_base_emissions,
        intervention_emissions.crop_project_emissions,
        intervention_emissions.crop_difference,
    )
    # =============================================================================

    # Fertilizer Emissions
    print_fertilizer_emissions(
        fertiliser_base_emissions=intervention_emissions.fertiliser_base_emissions,
        fertiliser_project_emissions=intervention_emissions.fertiliser_project_emissions,
        fertiliser_difference=intervention_emissions.fertiliser_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Litter Emissions
    print_litter_emissions(
        litter_base_emissions=intervention_emissions.litter_base_emissions,
        litter_project_emissions=intervention_emissions.litter_project_emissions,
        litter_difference=intervention_emissions.litter_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Fire Emissions
    print_fire_emissions(
        fire_base_emissions=intervention_emissions.fire_base_emissions,
        fire_project_emissions=intervention_emissions.fire_project_emissions,
        fire_difference=intervention_emissions.fire_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Tree Eemissions
    print_tree_emissions(
        tree_base_emissions=intervention_emissions.tree_base_emissions,
        tree_project_emissions=intervention_emissions.tree_project_emissions,
        tree_difference=intervention_emissions.tree_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Soil Emissions
    print_soil_emissions(
        soil_base_emissions=intervention_emissions.soil_base_emissions,
        soil_project_emissions=intervention_emissions.soil_project_emissions,
        soil_difference=intervention_emissions.soil_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Total Emissions
    emit_difference = (
        intervention_emissions.emit_project_emissions
        - intervention_emissions.emit_base_emissions
    )

    print_total_emissions(
        emit_base_emissions=intervention_emissions.emit_base_emissions,
        emit_project_emissions=intervention_emissions.emit_project_emissions,
        emit_difference=emit_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Summary of GHG pools
    summary_difference_data = [
        ["Difference Type", "Value", "Units"],
        [
            "Total Crop Difference",
            f"{sum(intervention_emissions.crop_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Fertiliser Difference",
            f"{sum(intervention_emissions.fertiliser_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Litter Difference",
            f"{sum(intervention_emissions.litter_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Fire Difference",
            f"{sum(intervention_emissions.fire_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Tree Difference",
            f"{sum(intervention_emissions.tree_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Soil Difference",
            f"{sum(intervention_emissions.soil_difference):.2f}",
            "t CO2 ha^-1",
        ],
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

    Climate.save(intervention_emissions.climate, plot_name + "_climate.csv")

    SoilParams.save(intervention_emissions.soil, plot_name + "_soil.csv")

    save_tree_growths(intervention_emissions.tree_growths, plot_name)

    save_tree_projects(intervention_emissions.tree_projects, plot_name=plot_name)

    save_crop_data(
        intervention_emissions.crop_base,
        intervention_emissions.crop_project,
        plot_name,
        "crop_model",
    )
    save_crop_data(
        intervention_emissions.crop_par_base,
        intervention_emissions.crop_par_project,
        plot_name,
        "crop_params",
    )

    InverseRothC.save(intervention_emissions.inverse_roth, plot_name + "_invRoth.csv")
    ForwardRothC.save(
        forward_roth_c=intervention_emissions.for_roth,
        no_of_years=N_YEARS,
        file=plot_name + "_forRoth.csv",
    )

    ForwardRothC.save(
        forward_roth_c=intervention_emissions.roth_base,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_base.csv",
    )
    ForwardRothC.save(
        forward_roth_c=intervention_emissions.roth_project,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_proj.csv",
    )

    Emit.save(
        intervention_emissions.emit_base_emissions,
        intervention_emissions.emit_project_emissions,
        plot_name + "_emit_proj.csv",
    )

    data = {
        "emit_base_emissions": intervention_emissions.emit_base_emissions,
        "emit_project_emissions": intervention_emissions.emit_project_emissions,
        "emit_difference": emit_difference,
        "soil_base_emissions": intervention_emissions.soil_base_emissions,
        "soil_project_emissions": intervention_emissions.soil_project_emissions,
        "soil_difference": intervention_emissions.soil_difference,
        "tree_base_emissions": intervention_emissions.tree_base_emissions,
        "tree_project_emissions": intervention_emissions.tree_project_emissions,
        "tree_difference": intervention_emissions.tree_difference,
        "fire_base_emissions": intervention_emissions.fire_base_emissions,
        "fire_project_emissions": intervention_emissions.fire_project_emissions,
        "fire_difference": intervention_emissions.fire_difference,
        "litter_base_emissions": intervention_emissions.litter_base_emissions,
        "litter_project_emissions": intervention_emissions.litter_project_emissions,
        "litter_difference": intervention_emissions.litter_difference,
        "fertiliser_base_emissions": intervention_emissions.fertiliser_base_emissions,
        "fertiliser_project_emissions": intervention_emissions.fertiliser_project_emissions,
        "fertiliser_difference": intervention_emissions.fertiliser_difference,
        "crop_base_emissions": intervention_emissions.crop_base_emissions,
        "crop_project_emissions": intervention_emissions.crop_project_emissions,
        "crop_difference": intervention_emissions.crop_difference,
    }

    write_emissions_csv(configuration, mod_run, n, st, data)

    # Plot stuff
    plot_tree_projects(intervention_emissions.tree_projects, plot_name)

    plt.close()

    ForwardRothC.plot(
        intervention_emissions.for_roth,
        no_of_years=N_YEARS,
        legend_string="initialisation",
    )

    ForwardRothC.plot(
        intervention_emissions.roth_base, no_of_years=N_YEARS, legend_string="baseline"
    )

    ForwardRothC.plot(
        intervention_emissions.roth_project,
        no_of_years=N_YEARS,
        legend_string="project",
        save_name=plot_name + "_soilModel.png",
    )
    plt.close()

    Emit.plot(intervention_emissions.emit_base_emissions, legend_string="baseline")
    Emit.plot(intervention_emissions.emit_project_emissions, legend_string="project")

    # TODO: what is this? How could `ax` be attached to Emission?
    # Why is it done here instead of in the plot function?
    # Commenting out for now
    # emit.Emission.ax.plot(emit_difference, label="difference")
    # emit.Emission.ax.legend(loc="best")

    plt.savefig(os.path.join(configuration.OUTPUT_DIR, plot_name + "_emissions.png"))
    plt.close()

    Emit.save(
        emit_base_emissions=intervention_emissions.emit_base_emissions,
        emit_proj_emissions=intervention_emissions.emit_project_emissions,
        file=plot_name + "_emissions.csv",
    )

    return (
        sum(intervention_emissions.crop_difference),
        sum(intervention_emissions.fertiliser_difference),
        sum(intervention_emissions.litter_difference),
        sum(intervention_emissions.fire_difference),
        sum(intervention_emissions.tree_difference),
        sum(intervention_emissions.soil_difference),
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
