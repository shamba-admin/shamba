#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import os
import logging as log
import argparse

from shamba.model import configuration

def get_command_line_arguments():
    """
    Parse the command line arguments for graph and report generation 
    (-g, -r) and verbosity of output (-v=info,-vv=debug). 
    Return arguments.
    """

    parser = argparse.ArgumentParser()
    verboseMsg = "Set verbosity level (e.g. -v -v or vv more verbose than -v)"
    parser.add_argument(
            "-v", "--verbose", action="count", dest="verbose",
            default=0, help=verboseMsg
    )
    parser.add_argument(
            "-p", "--param", action="store", dest="param")
    parser.add_argument(
            "-r", "--report", action="store_true", dest="report",
            default=False, help="Print report to stdout"
    )
    parser.add_argument(
            "-g", "--graph", action="store_true", dest="graph",
            default=False, help="Show plots"
    )

    arguments = parser.parse_args()

    # Set up logging level based on v arguments
    if arguments.verbose == 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif arguments.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    elif arguments.verbose == 0:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return arguments

def print_metadata():
    """
    Print the project metadata (timestamp and unique hex ID)
    calculated in the cfg module.
    
    """
    filepath = os.path.join(configuration.SAV_DIR, '.info')
    with open(filepath, 'w') as f:
        id_str = configuration.ID if configuration.ID is not None else ''
        time_str = configuration.TIME if configuration.TIME is not None else ''
        proj_name_str = configuration.PROJ_NAME if configuration.PROJ_NAME is not None else ''
        f.write(f"{id_str}\n{time_str}\n{proj_name_str}\n\n")


