#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import logging as log
import os
from datetime import datetime
import questionary
from questionary import Validator, ValidationError
from questionary import Choice

from model import configuration
import model.tree_growth as TreeGrowth
from model.soil_models.soil_model_types import SoilModelType
from model.common.constants import (
    DEFAULT_USE_API,
    DEFAULT_ALLOMORPHY,
    DEFAULT_GWP,
    GWP_list,
)
from model.common.validations import validate_integer, validate_numerical


def get_arguments_interactively():
    """
    Prompt the user for arguments interactively using the `questionary` library.
    Return a dictionary containing the argument values.
    """
    arguments = {}

    # Display instructions using a pure print — not necessary to prompt here
    print(
        """
INSTRUCTIONS

Complete in full the Excel worksheet 'SHAMBA input output template v1.2',
(located in the 'data-input-templates' folder)    
including all references for information. The reviewer will reject the
modelling unless it is fully referenced. See the instructions in the Excel
worksheet.

On the '_questionnaire' worksheet, you must enter a value in each of the
blue cells in the 'Input data' column (column N) in response to each 
'data collection question', otherwise the model will not run properly. 
If the question is not relevant to the land use you are modelling, enter zero.

To run the model for a particular intervention, save the relevant 
'_input.csv' file into the new shamba/projects/"project-name"/input
folder, along with other input files needed. This is the 'source directory'
you must specify when prompted at the command line.'

If nitrogen allocations, carbon, root/shoot and/or wood density attributes
differ between tree cohorts, add a new row specifying these tree parameters
to the tree_defaults.csv in the shamba/default_input folder and make sure the 
'_input.csv' file correctly attributes each tree cohort to the relevant 
parameters under 'trees in baseline' and 'trees in project'.

If allometric functions not included in the SHAMBA code base are to be used, 
write these in a python file named 'project_allometry.py' in your source directory. 
Ensure:
1. each function returns aboveground biomass in kg C for a single tree size;
    using `tree_params.carbon` where necessary.
2. the file includes a dictionary called 'allometric' matching each allometric 
    function to a key, so that you can select it at the command line. 
Note: functions using input data other than diameter at breast height 
(dbh) will need careful handling. A suggestion of how to handle this is included
in the example project (/projects/examples/UG_TS_2016/input/project_allometry.py)

Soil and climate data is either sourced from APIs, or from local csv files of your
own data. To use your own values for soil and climate data, csv files should
be added to the source directory (alongisde your input file). 
The climate data csv must be called climate.csv and match the format shown in 
/projects/examples/UG_TS_2016/input/climate.csv.
The soil data csv must be called soil-info.csv and match the format shown in
/projects/examples/UG_TS_2016/input/soil-info.csv.
        """
    )

    # Generate timestamp for default project name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prompt for project name
    project_name = questionary.text(
        "Enter project name (or use auto-generated name)",
        default=f"project_{timestamp}",
    ).ask()
    arguments["project-name"] = project_name

    # Prompt for source directory
    source_directory = questionary.text(
        "Enter source directory path relative to /projects/" " (or use example)",
        default=f"examples/UG_TS_2016/input",
    ).ask()
    arguments["source-directory"] = source_directory

    # Prompt for use-api (boolean)
    use_api = questionary.confirm("Use API for climate and soil data?", default=DEFAULT_USE_API).ask()
    arguments["use-api"] = use_api

    # Prompt for n_cohorts
    # Default to 1 if integer not provided
    n_cohorts = questionary.text("Enter number of tree cohorts (defaults to 1): ", validate=validate_integer, default="1").ask()
    arguments["n-cohorts"] = int(n_cohorts)

    # Prompt for allometric key list
    own_allometry = questionary.confirm(
        "Do you have allometric functions to use that are not in SHAMBA's default list? (if yes, please see instructions):", default=False).ask()
    own_allometric_keys = []
    allometric_keys = list(TreeGrowth.allometric.keys())
    if own_allometry == True:
        import sys
        import importlib
        
        source_dir = os.path.join(configuration.PROJECT_DIR, arguments["source-directory"])
        sys.path.insert(0, source_dir)
        project_allometry = importlib.import_module('project_allometry')
        own_allometric_keys = list(project_allometry.allometric.keys())
        
    all_allometric_keys = allometric_keys + own_allometric_keys


    # Prompt for allometric key, cohort by cohort
    cohort_allometric_keys = []

    base_selected_allometric_key = questionary.select(
        "Select an Allometric Key for the baseline species:", choices=all_allometric_keys, default=DEFAULT_ALLOMORPHY
        ).ask()
    
    cohort_allometric_keys.append(base_selected_allometric_key)

    for i in range(int(n_cohorts)):
        selected_allometric_key = questionary.select(
        "Select an Allometric Key for each species in the cohort, in the same order as the input file:", 
        choices=all_allometric_keys, default=DEFAULT_ALLOMORPHY).ask()
        cohort_allometric_keys.append(selected_allometric_key)
    arguments["allometric-keys"] = cohort_allometric_keys

    # Prompt for GWP
    gwp_keys = list(GWP_list.keys())
    selected_gwp_key = questionary.select(
        "Select Global Warming Potential values:", choices=gwp_keys, default=DEFAULT_GWP
    ).ask()
    arguments["gwp"] = GWP_list[selected_gwp_key]

    # Prompt for soil model
    soil_models = [
        Choice(title="Roth C", value=SoilModelType.ROTH_C),
        Choice(title="Example Soil Model", value=SoilModelType.EXAMPLE),
    ]

    # selected_soil_model = questionary.select(
    #     "Select a soil model:",
    #     choices=soil_models,
    #     default=SoilModelType.ROTH_C
    # ).ask()
    arguments["soil-model"] = SoilModelType.ROTH_C

    # Prompt for whether to print to stdout
    print_to_stdout = questionary.confirm("Results will be saved to csv files. Do you also want to print all to stdout?", default=False).ask()
    arguments["print-to-stdout"] = print_to_stdout

    # Prompt for input file name with default
    input_file_name = questionary.text(
        "Enter the name of the input file:", default="WL_input.csv"
    ).ask()
    arguments["input-file-name"] = input_file_name

    # Prompt for output title
    output_title = questionary.text(
        "Enter the title of the output file:", default="WL"
    ).ask()
    arguments["output-title"] = output_title

    # Set logging configuration
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    return arguments
