#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import logging as log
import os
from datetime import datetime
import questionary
from questionary import Validator, ValidationError

from model import configuration
import model.tree_growth as TreeGrowth


def get_arguments_interactively():
    """
    Prompt the user for arguments interactively using the `questionary` library.
    Return a dictionary containing the argument values.
    """
    arguments = {}

    # Generate timestamp for default project name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prompt for project name
    project_name = questionary.text(
        "Enter project name (or leave blank to auto-generate)",
        default=f"project_{timestamp}"
    ).ask()
    arguments["project-name"] = project_name

    # Prompt for use-api (boolean)
    use_api = questionary.confirm("Use API?", default=False).ask()
    arguments["use-api"] = use_api

    # Prompt for allometric key
    allometric_keys = list(TreeGrowth.allometric.keys())
    selected_allometric_key = questionary.select(
        "Select an Allometric Key:",
        choices=allometric_keys,
        default="chave dry"
    ).ask()
    arguments["allometric-key"] = selected_allometric_key

    # Prompt for whether to print to stdout
    print_to_stdout = questionary.confirm("Print to stdout?", default=False).ask()
    arguments["print-to-stdout"] = print_to_stdout

    # Display instructions using a pure print — not necessary to prompt here
    print(
        """
INSTRUCTIONS

Complete in full the Excel worksheet 'SHAMBA input output template v1',
(located in the 'data-input-templates' folder)    
including all references for information. The reviewer will reject the
modelling unless it is fully referenced. See the instructions in the Excel
worksheet.

On the '_questionnaire' worksheet, you must enter a value in each of the
blue cells in the 'Input data' column (column N) in response to each 
'data collection question'. Otherwise the model will not run properly. 
If the question is not relevant to the land use you are modelling, enter zero.

To run the model for a particular intervention, save the relevant 
'_input.csv' file into the new shamba/projects/"project-name"/input
folder. 

If nitrogen allocations, carbon, root/shoot and/or wood density attributes
differ between tree cohorts, add a new row specifying these tree parametres
to the tree_defaults.csv in the shamba/default_input folder and make sure the 
'_input.csv' file correctly attributes each tree cohort to the relevant 
parametres under 'trees in baseline' and 'trees in project'.
        """
    )

    # Prompt for input file name with default
    input_file_name = questionary.text(
        "Enter the name of the input file:",
        default="WL_input.csv"
    ).ask()
    arguments["input-file-name"] = input_file_name

    # Prompt for output title
    output_title = questionary.text(
        "Enter the title of the output file:",
        default="WL"
    ).ask()
    arguments["output-title"] = output_title

    # Set logging configuration
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    return arguments
