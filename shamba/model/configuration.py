#!/usr/bin/py

"""Module for global variables for all of SHAMBA."""
import os
import uuid
from time import gmtime, strftime

# input and output files for specific project
# change this for specific projects
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(script_dir, "..")
PROJECT_DIR = os.path.join(BASE_PATH, "projects")
TESTS_DIR = os.path.join(BASE_PATH, "tests")
# no need to specify SAV, INP, OUT directories if specified in cl file
SAVE_DIR = os.path.join(PROJECT_DIR, "default")  # overwrite this later
INPUT_DIR = os.path.join(SAVE_DIR, "input")
OUTPUT_DIR = os.path.join(SAVE_DIR, "output")

# Number of years for model to run and accounting period
# NOTE: change this in main after tree max age data is read
# no need to specify years if specified in cl file
N_YEARS = 30
N_ACCT = 30


# Save the time (that cfg is imported) and generate
# universally unique identifier (uuid) for the project

# convert seconds since the epoch to an ISO-8601 time in UTC
TIME = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
ID = uuid.uuid4().hex
PROJ_NAME = None
