#!/usr/bin/python

"""Module containing Emission class and reduce_from_fire function."""

import logging as log
import os

import matplotlib.pyplot as plt
import numpy as np

from model import configuration
from model.common import csv_handler
from model.common.constants import (ef, N_ef, DEFAULT_GWP, cf, C_to_CO2_conversion_factor, N_to_N2O_conversion_factor, volatile_frac_org, volatile_frac_synth)

# Fire vector - can redefine from elsewhere if there are fires
# fire = np.zeros(configuration.N_YEARS)

# Reduce crop/tree/litter outputs due to fire
def reduce_from_fire(
    no_of_years, crop=[], tree=[], litter=[], fire=[], output_type="carbon"
):
    """
    Calculate the crop and tree outputs
    of specified type (e.g. 'carbon', 'nitrogen', 'DMon', DMoff')
    after having their mass 'reduced' (burned) by fire.

    Args:
        crop: list of crop objects
        tree: list of tree objects
        litter: list of litter objects
        output_type: type of output from crop,tree,litter to use
                    (i.e. 'carbon, 'nitrogen', 'DMoff','DMon')
    Returns:
        reduced: total of above and below for crop and tree (in a duple)

    """
    # Add up all inputs
    crop_inputs = {
        "above": np.zeros(no_of_years),
        "below": np.zeros(no_of_years),
    }
    tree_inputs = {
        "above": np.zeros(no_of_years),
        "below": np.zeros(no_of_years),
    }

    for s in ["above", "below"]:
        try:
            for c in crop:
                crop_inputs[s] += c.output[s][output_type]
            for t in tree:
                tree_inputs[s] += t.output[s][output_type]
            for li in litter:
                tree_inputs[s] += li.output[s][output_type]
        except KeyError:
            log.exception("Invalude output_type parameter in reduce_from_fire")

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
    no_of_years,
    forward_soil_model=None,
    crop=[],
    tree=[],
    litter=[],
    fert=[],
    fire=[],
    burn_off=True,
) -> np.ndarray:
    """Create an array.
    Optional arguments gives flexibility about what/what kind of
    emissions to calculate

    Args:
        forward_soil_model: ForwardSoilModelData object
        crop: list of crop objects
        tree: list of tree objects
        litter: list of litter objects
        fert: list of litter objects for synthetic fert
        burn_off: whether off-farm crop residues are burned
                    can be simply True/False (all residues are burned or not)
                    or a list of bools (corresponding to each crop in
                    crop list)

    """

    # Calculate total emission (for types that aren't None or empty)
    emissions = np.zeros(no_of_years)
    # += the sources (nitrogen, fire, fertiliser)
    # and -= the sinks (biomass, soil)

    emissions_soc = -soc_sink(forward_soil_model, no_of_years) if forward_soil_model is not None else 0
    emissions_tree = -tree_sink(tree, no_of_years) if tree else 0
    emissions_nitro = (
        nitrogen_emit(no_of_years=no_of_years, crop=crop, tree=tree, litter=litter)
        if (crop or tree or litter)
        else 0
    )
    emissions_fire = (
        fire_emit(crop, tree, litter, fire, no_of_years, burn_off=burn_off)
        if (crop or tree or litter)
        else 0
    )
    emissions_fert = fert_emit(litter, fert, no_of_years) if (fert or litter) else 0

    total_emissions = (
        emissions
        + emissions_soc
        + emissions_tree
        + emissions_nitro
        + emissions_fire
        + emissions_fert
    )

    # We only care about portion in the project accounting period
    return total_emissions[0:no_of_years]


def plot(emissions, legend_string, save_name=None):
    """Plot total carbon vs year for emissions.

    Args:
        legend_string: string to put in legend

    """
    fig = plt.figure()
    fig.suptitle("Emissions")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Emissions vs time")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Emissions (t CO2 ha^-1)")

    ax.plot(emissions, label=legend_string)
    ax.legend(loc="best")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))


def save(emit_base_emissions, emit_proj_emissions=None, file="emissions.csv"):
    """Save emission data to csv file. Default path is OUTPUT_DIR.

    Args:
        emit_base_emissions: Emission object to print.
                    If two emissions are being compared (base and proj),
                    this should be the baseline object.
        emit_proj_emissions: Second emission object to be compared - difference
                    is emit_proj - emit_base, so ensure correct order
        file: filename or path to csv file

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


def soc_sink(forward_soil_model, no_of_years):
    """
    Calculate SOC differences from year to year (carbon sink)
    Instance of ForwardRothC is the argument.

    Return vector with differences.
    """
    # total of all pools
    soc = np.sum(forward_soil_model.SOC, axis=1)

    # IMPORTANT: make sure soc is of length N_YEARS+1
    # (N years, inclusive of beginning and end = N+1 array entries)
    delta_SOC = np.zeros(no_of_years)

    for i in range(no_of_years):
        delta_SOC[i] = (soc[i + 1] - soc[i]) * C_to_CO2_conversion_factor

    return delta_SOC


def tree_sink(tree, no_of_years):
    """
    Calculate woody biomass pool sizes from year to year (carbon sink)
    Tree object is the input argument

    Return vector with differences.
    """

    biomass = np.zeros(no_of_years + 1)
    for t in tree:
        biomass += np.sum(t.woody_biomass, axis=1)

    delta = np.zeros(no_of_years)
    for i in range(no_of_years):
        delta[i] = (biomass[i + 1] - biomass[i]) * C_to_CO2_conversion_factor

    return delta


def nitrogen_emit(no_of_years, crop, tree, litter):
    """
    Calculate and return emissions due to nitrogen.
    crop_out == list of output dicts from crops
    tree_out == list of output dicts from trees
    litter_out == list of output dicts from litter
    """
    toEmit_crop, toEmit_tree = reduce_from_fire(
        no_of_years=no_of_years,
        crop=crop,
        tree=tree,
        litter=litter,
        output_type="nitrogen",
    )
    to_emit = toEmit_crop + toEmit_tree



    return to_emit * N_ef * N_to_N2O_conversion_factor * DEFAULT_GWP["N2O"]


def fire_emit(crop, tree, litter, fire, no_of_years, burn_off=True):
    """Calculate and return emissions due to fire.
    crop: list of crop models
    tree: list of tree models
    litter: list of litter models
    burn_off == True if all removed crop residues are burned every year
                Can also be a list corresponding to each crop
                    e.g. [True, False] -> burn crop1 off-res but not crop2
    """

    emit = np.zeros(no_of_years)
    # sum up the above-ground mass on-farm eligible to be burned
    # off-farm is summed up if any of the crops has burn=True
    crop_inputs_on = np.zeros(no_of_years)
    tree_inputs_on = np.zeros(no_of_years)
    for c in crop:
        crop_inputs_on += c.output["above"]["DMon"]
    for t in tree:
        tree_inputs_on += t.output["above"]["DMon"]
    for li in litter:
        tree_inputs_on += li.output["above"]["DMon"]

    crop_temperature = ef["crop_CH4"] * DEFAULT_GWP["CH4"] + ef["crop_N2O"] * DEFAULT_GWP["N2O"]
    tree_temperature = ef["tree_CH4"] * DEFAULT_GWP["CH4"] + ef["tree_N2O"] * DEFAULT_GWP["N2O"]

    # Burned when fire == 1
    emit += crop_inputs_on * fire * cf["crop"] * crop_temperature
    emit += tree_inputs_on * fire * cf["tree"] * tree_temperature

    # whether to burn off-farm crop residues every year
    # construct a list if only one bool is given
    if burn_off is True:
        burn_off_lst = len(crop) * [True]
    elif burn_off is False:
        burn_off_lst = len(crop) * [False]
    else:
        burn_off_lst = burn_off

    if any(burn_off_lst):
        crop_inputs_off = np.zeros(no_of_years)
        for i, c in enumerate(crop):
            if burn_off_lst[i]:
                crop_inputs_off += c.output["above"]["DMoff"]

        emit += crop_inputs_off * cf["crop"] * crop_temperature

    emit *= 0.001  # convert to tonnes
    return emit


def fert_emit(litter, fert, no_of_years):
    """Calculate and return emissions due to fertiliser use.
    Args:
        litter: list-like of litter model objects
        fert: list-like of fertiliser model object
                (special case of litter model object)
    """


    # calculate emissions
    emit = np.zeros(no_of_years)
    # still need to add fertiliser ************
    for li in litter:
        emit += np.array(li.output["above"]["nitrogen"], dtype=float) * (
            1 - volatile_frac_org
        )
    for f in fert:
        emit += np.array(f.output["above"]["nitrogen"], dtype=float) * (
            1 - volatile_frac_synth
        )

    emit *= N_ef * N_to_N2O_conversion_factor * DEFAULT_GWP["N2O"]

    return emit
