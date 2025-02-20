#!/usr/bin/python

"""Module containing Emission class and reduceFromFire function."""

import logging as log
import os
import numpy as np
import matplotlib.pyplot as plt

from model import configuration
from model.common import csv_handler

# Fire vector - can redefine from elsewhere if there are fires
fire = np.zeros(configuration.N_YEARS)

# Emissions stuff
# From table 2.5 IPCC 2006 GHG Inventory
ef = {"crop_N2O": 0.07, "crop_CH4": 2.7, "tree_N2O": 0.2, "tree_CH4": 6.8}

# Global warming potential from IPCC 2006 GHG Inventory
gwp = {"N2O": 310, "CH4": 21}

# combustion factor from IPCC AFOLU table
cf = {"crop": 0.8, "tree": 0.74}


# Reduce crop/tree/litter outputs due to fire
def reduceFromFire(crop=[], tree=[], litter=[], fire=[], outputType="carbon"):
    """
    Calculate the crop and tree outputs
    of specified type (e.g. 'carbon', 'nitrogen', 'DMon', DMoff')
    after having their mass 'reduced' (burned) by fire.

    Args:
        crop: list of crop objects
        tree: list of tree objects
        litter: list of litter objects
        outputType: type of output from crop,tree,litter to use
                    (i.e. 'carbon, 'nitrogen', 'DMoff','DMon')
    Returns:
        reduced: total of above and below for crop and tree (in a duple)

    """
    # Add up all inputs
    crop_inputs = {
        "above": np.zeros(configuration.N_YEARS),
        "below": np.zeros(configuration.N_YEARS),
    }
    tree_inputs = {
        "above": np.zeros(configuration.N_YEARS),
        "below": np.zeros(configuration.N_YEARS),
    }
    for s in ["above", "below"]:
        try:
            for c in crop:
                crop_inputs[s] += c.output[s][outputType]
            for t in tree:
                tree_inputs[s] += t.output[s][outputType]
            for li in litter:
                tree_inputs[s] += li.output[s][outputType]
        except KeyError:
            log.exception("Invalude outputType parameter in reduceFromFire")

    # Reduce above-ground inputs from fire
    for i in np.where(fire == 1):
        crop_inputs["above"][i] *= 1 - cf["crop"]
        tree_inputs["above"][i] *= 1 - cf["crop"]

    # Return sum of above and below
    reduced = (sum(crop_inputs.values()), sum(tree_inputs.values()))

    return reduced

"""
Emission class for calculating total GHG emissions
from soil, fire, fertiliser, and/or nitrogen.

Instance variables
------------------
emissions   vector of yearly GHG emissions in t CO2e/ha

"""

def create(
    forRothC=None, crop=[], tree=[], litter=[], fert=[], fire=[], burnOff=True
):
    """Initialise object.
    Optional arguments gives flexibility about what/what kind of
    emissions to calculate

    Args:
        forRothC: ForwardRothC object
        crop: list of crop objects
        tree: list of tree objects
        litter: list of litter objects
        fert: list of litter objects for synthetic fert
        burnOff: whether off-farm crop residues are burned
                    can be simply True/False (all residues are burned or not)
                    or a list of bools (corresponding to each crop in
                    crop list)

    """

    # Calculate total emission (for types that aren't None or empty)
    emissions = np.zeros(configuration.N_YEARS)
    # += the sources (nitrogen, fire, fertiliser)
    # and -= the sinks (biomass, soil)

    emissions_soc = -soc_sink(forRothC) if forRothC is not None else 0
    emissions_tree = -tree_sink(tree) if tree else 0
    emissions_nitro = nitrogen_emit(crop, tree, litter) if (crop or tree or litter) else 0
    emissions_fire = fire_emit(crop, tree, litter, fire, burn_off=burnOff) if (crop or tree or litter) else 0
    emissions_fert = fert_emit(litter, fert) if (fert or litter) else 0

    total_emissions = (
        emissions +
        emissions_soc +
        emissions_tree +
        emissions_nitro +
        emissions_fire +
        emissions_fert
    )

    # We only care about portion in the project accounting period
    return total_emissions[0 : configuration.N_ACCT]

def plot(emissions, legendStr, saveName=None):
    """Plot total carbon vs year for emissions.

    Args:
        legendStr: string to put in legend

    """
    fig = plt.figure()
    fig.suptitle("Emissions")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Emissions vs time")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Emissions (t CO2 ha^-1)")

    ax.plot(emissions, label=legendStr)
    ax.legend(loc="best")

    if saveName is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

def save(emit_base_emissions, emit_proj_emissions=None, file="emissions.csv"):
    """Save emission data to csv file. Default path is OUTPUT_DIR.

    Args:
        emit_base_emissions: Emission object to print.
                    If two emissions are being compared (base and proj),
                    this should be the baseline object.
        emit_proj_emissions: Second emission object to be compared - difference
                    is emit_proj - emit_base, so ensure correct order
        file: filename or path to csv file
        printTotal: print total of each column
                    **currently only works if emit_proj not given**

    """
    if emit_proj_emissions is None:
        data = emit_base_emissions  # just one emission vector to save
        cols = ["emissions"]
    else:
        data = np.column_stack(
            (
                emit_base_emissions,
                emit_proj_emissions,
                emit_proj_emissions - emit_base_emissions,
            )
        )
        cols = ["baseline", "project", "difference"]

    csv_handler.print_csv(file, data, col_names=cols, print_column=True)

def soc_sink(forRothC):
    """
    Calculate SOC differences from year to year (carbon sink)
    Instance of ForwardRothC is the argument.

    Return vector with differences.
    """
    # total of all pools
    soc = np.sum(forRothC.SOC, axis=1)

    # To convert from [t C/ha] to [t CO2/ha]
    conversionFactor = 44.0 / 12

    # IMPORTANT: make sure soc is of length N_YEARS+1
    # (N years, inclusive of beginning and end = N+1 array entries)
    deltaSOC = np.zeros(configuration.N_YEARS)

    for i in range(configuration.N_YEARS):
        deltaSOC[i] = (soc[i + 1] - soc[i]) * conversionFactor

    return deltaSOC

def tree_sink(tree):
    """
    Calculate woody biomass pool sizes from year to year (carbon sink)
    Tree object is the input argument

    Return vector with differences.
    """
    conversionFactor = 44.0 / 12

    biomass = np.zeros(configuration.N_YEARS + 1)
    for t in tree:
        biomass += np.sum(t.woodyBiom, axis=1)

    delta = np.zeros(configuration.N_YEARS)
    for i in range(configuration.N_YEARS):
        delta[i] = (biomass[i + 1] - biomass[i]) * conversionFactor

    return delta

def nitrogen_emit(crop, tree, litter):
    """
    Calculate and return emissions due to nitrogen.
    crop_out == list of output dicts from crops
    tree_out == list of output dicts from trees
    litter_out == list of output dicts from litter
    """
    toEmit_crop, toEmit_tree = reduceFromFire(
        crop, tree, litter, outputType="nitrogen"
    )
    toEmit = toEmit_crop + toEmit_tree

    ef = 0.01  # emission factor [kbN20-N/kg N]
    mw = 44.0 / 28  # for N2O-N to N2O

    return toEmit * ef * mw * gwp["N2O"]

def fire_emit(crop, tree, litter, fire, burn_off=True):
    """Calculate and return emissions due to fire.
    crop: list of crop models
    tree: list of tree models
    litter: list of litter models
    burn_off == True if all removed crop residues are burned every year
                Can also be a list corresponding to each crop
                    e.g. [True, False] -> burn crop1 off-res but not crop2
    """

    emit = np.zeros(configuration.N_YEARS)
    # sum up the above-ground mass on-farm eligible to be burned
    # off-farm is summed up if any of the crops has burn=True
    crop_inputs_on = np.zeros(configuration.N_YEARS)
    tree_inputs_on = np.zeros(configuration.N_YEARS)
    for c in crop:
        crop_inputs_on += c.output["above"]["DMon"]
    for t in tree:
        tree_inputs_on += t.output["above"]["DMon"]
    for li in litter:
        tree_inputs_on += li.output["above"]["DMon"]

    cropTemp = ef["crop_CH4"] * gwp["CH4"] + ef["crop_N2O"] * gwp["N2O"]
    treeTemp = ef["tree_CH4"] * gwp["CH4"] + ef["tree_N2O"] * gwp["N2O"]

    # Burned when fire == 1
    emit += crop_inputs_on * fire * cf["crop"] * cropTemp
    emit += tree_inputs_on * fire * cf["tree"] * treeTemp

    # whether to burn off-farm crop residues every year
    # construct a list if only one bool is given
    if burn_off is True:
        burn_off_lst = len(crop) * [True]
    elif burn_off is False:
        burn_off_lst = len(crop) * [False]
    else:
        burn_off_lst = burn_off

    if any(burn_off_lst):
        crop_inputs_off = np.zeros(configuration.N_YEARS)
        for i, c in enumerate(crop):
            if burn_off_lst[i]:
                crop_inputs_off += c.output["above"]["DMoff"]

        emit += crop_inputs_off * cf["crop"] * cropTemp

    emit *= 0.001  # convert to tonnes
    return emit

def fert_emit(litter, fert):
    """Calculate and return emissions due to fertiliser use.
    Args:
        litter: list-like of litter model objects
        fert: list-like of fertiliser model object
                (special case of litter model object)
    """
    # Some parameters. See methodology
    ef = 0.01
    mw_ratio = 44.0 / 28
    gwp = 310.0
    volatile_frac_synth = 0.1
    volatile_frac_org = 0.2

    # calculate emissions
    emit = np.zeros(configuration.N_YEARS)
    # still need to add fertiliser ************
    for li in litter:
        emit += li.output["above"]["nitrogen"] * (1 - volatile_frac_org)
    for f in fert:
        emit += f.output["above"]["nitrogen"] * (1 - volatile_frac_synth)

    emit *= ef * mw_ratio * gwp

    return emit
