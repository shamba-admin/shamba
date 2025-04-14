#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import logging as log
import os
from datetime import datetime

from model import configuration
from model.tree_growth import allometric


def get_arguments_interactively():
    """
    Prompt the user for arguments interactively.
    Return a dictionary containing the argument values.
    """
    arguments = {}

    # Prompt for param
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = input("Enter project name (or press Enter to skip): ").strip()
    arguments["project-name"] = project_name if project_name else f"project_{timestamp}"

    # Display allometric options
    allometric_keys = list(allometric.keys())
    print("\nSelect Allometric Key:")
    for idx, key in enumerate(allometric_keys):
        print(f"{idx + 1}: {key}")

    # Get user input for allometric choice
    try:
        allometric_choice = (
            int(input("Choose an allometric option by number: ").strip()) - 1
        )
    except ValueError:
        allometric_choice = -1  # A value that will trigger the default assignment

    # Validate the choice and assign the corresponding key
    if allometric_choice >= 0 and allometric_choice < len(allometric_keys):
        selected_allometric_key = allometric_keys[allometric_choice]
        arguments["allometric-key"] = selected_allometric_key
    else:
        print("Invalid choice. Defaulting to 'chave dry'.")
        arguments["allometric-key"] = "chave dry"  # Default value if choice is invalid

    # Add some instructions here
    print(
        """
    INSTRUCTIONS
        
    Complete in full the Excel worksheet 'SHAMBA input output template v1',
    (located in the 'data-input-templates' folder)    
    including all references for information. The reviewer will reject the
    modelling unless it is fully referenced. See the instructions in the Excel
    worksheet.
    
    On the '_questionnaire' worksheet, you must enter a value in each of the
    blue cells in  the 'Input data' column (column N) in response to each 
    'data collection question'. Otherwise the model will not run properly. 
    If the question is not relevent to the land use you are modelling, enter zero.
    
    To run the model for a particular intervention, save the relevant 
    '_input.csv' file into the new shamba/projects/"project-name"/input
    folder. 

    If nitrogen allocations, carbon, root/shoot and/or wood density attributes
    differ between tree cohorts, add a new row specifying these tree parametres
    to the the tree_defaults.csv at shamba/default_input folder and make sure the 
    '_input.csv' file correctly attributes each tree cohort to the relevant 
    parametres under 'trees in baseline' and 'trees in project'
    """
    )

    input_file_name = input("Enter the name of the input file: ").strip()
    arguments["input-file-name"] = (
        input_file_name if input_file_name else "WL_input.csv"
    )

    output_title = input("Enter the title of the output file (default is 'WL'): ")
    arguments["output-title"] = output_title if output_title else "WL"

    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)  # Or DEBUG

    return arguments


def print_metadata():
    """
    Print the project metadata (timestamp and unique hex ID)
    calculated in the cfg module.

    """
    filepath = os.path.join(configuration.SAVE_DIR, ".info")
    with open(filepath, "w") as f:
        id_str = configuration.ID if configuration.ID is not None else ""
        time_str = configuration.TIME if configuration.TIME is not None else ""
        proj_name_str = (
            configuration.PROJ_NAME if configuration.PROJ_NAME is not None else ""
        )
        f.write(f"{id_str}\n{time_str}\n{proj_name_str}\n\n")
