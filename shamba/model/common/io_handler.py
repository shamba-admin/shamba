#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import os
import logging as log
from datetime import datetime


from model import configuration


def get_arguments_interactively():
    """
    Prompt the user for arguments interactively.
    Return a dictionary containing the argument values.
    """
    arguments = {}

    # Prompt for verbosity
    print("Verbosity level:")
    print("0 - Normal")
    print("1 - Info")
    print("2 - Debug")
    arguments["verbose"] = int(
        input("Enter verbosity level (0/1/2): ").lower().startswith("0")
    )

    # Prompt for param
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = input("Enter project name (or press Enter to skip): ").strip()
    arguments["project-name"] = project_name if project_name else f"project_{timestamp}"

    # Prompt for report
    arguments["report"] = (
        input("Print report to stdout? (y/n): ").lower().startswith("y")
    )

    # Prompt for graph
    arguments["graph"] = input("Show plots? (y/n): ").lower().startswith("y")

    # Add some instructions here
    print(
        """Complete in full the Excel worksheet 'SHAMBA input output template v1',
    (located in the 'data-input-templates' folder)    
    including all references for information. The reviewer will reject the
    modelling unless it is fully referenced. See the instructions in the Excel
    worksheet.
    
    On the '_questionnaire' worksheet, you must enter a value in each of the
    blue cells in  the 'Input data' column (column N) in response to each 
    'data collection question'. Otherwise the model will not run properly. 
    If the question is not relevent to the land use you are modelling, enter zero."""
    )

    print(
        """
    To run the model for a particular intervention, save the relevant 
    '_input.csv' file into the new shamba/projects/"project-name"/input
    folder.   
    """
    )

    input_file_name = input("Enter the name of the input file: ").strip()
    arguments["input-file-name"] = (
        input_file_name if input_file_name else "WL_input.csv"
    )

    print(
        """
    If nitrogen allocations, carbon, root/shoot and/or wood density attributes
    differ between tree cohorts, add a new row specifying these tree parametres
    to the the tree_defaults.csv at shamba/default_input folder and make sure the 
    '_input.csv' file correctly attributes each tree cohort to the relevant 
    parametres under 'trees in baseline' and 'trees in project'
    """
    )

    output_title = input("Enter the title of the output file (default is 'WL'): ")
    arguments["output-title"] = output_title if output_title else "WL"

    # Set up logging level based on verbosity
    if arguments["verbose"] == 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif arguments["verbose"] == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

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
