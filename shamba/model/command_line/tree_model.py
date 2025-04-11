#!/usr/bin/python

"""Module containing Tree class."""

import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from marshmallow import Schema, fields, post_load

from .. import configuration
from ..common import csv_handler
from .common_schema import OutputSchema as ClimateDataOutputSchema
from .tree_growth import TreeGrowthSchema, derivative_functions
from .tree_params import ROOT_IN_TOP_30, TreeParamsSchema

"""
Tree model class. Calculate residues and soil inputs
for given params and growth.

Instance variables
----------------
tree_params     TreeParams object with the params (dens, carbon, etc.)
growth          TreeGrowth object governing growth of trees
output          output to soil,fire in t C ha^-1
                    (dict with keys 'carbon,'nitrogen','DMon','DMoff')
woody_biomass       vector of woody biomass in each pool in t C ha^-1

"""


class MassBalanceData(Schema):
    in_ = fields.List(fields.Float(allow_nan=True), required=True)
    out = fields.List(fields.Float(allow_nan=True), required=True)
    acc = fields.List(fields.Float(allow_nan=True), required=True)
    bal = fields.List(fields.Float(allow_nan=True), required=True)


class TreeModel:
    def __init__(
        self,
        tree_params,
        tree_growth,
        alloc,
        turnover,
        thin_frac,
        mort_frac,
        thin,
        mort,
        output,
        woody_biomass,
        balance,
    ):
        self.tree_params = tree_params
        self.tree_growth = tree_growth
        self.alloc = alloc
        self.turnover = turnover
        self.thin_frac = thin_frac
        self.mort_frac = mort_frac
        self.thin = thin
        self.mort = mort
        self.output = output
        self.woody_biomass = np.array(woody_biomass)
        self.balance = balance


class TreeModelSchema(Schema):
    tree_params = fields.Nested(TreeParamsSchema, required=True)
    tree_growth = fields.Nested(TreeGrowthSchema, required=True)
    alloc = fields.List(fields.Float, required=True)
    turnover = fields.List(fields.Float, required=True)
    thin_frac = fields.List(fields.Float, required=True)
    mort_frac = fields.List(fields.Float, required=True)
    thin = fields.List(fields.Float, required=True)
    mort = fields.List(fields.Float, required=True)
    output = fields.Nested(ClimateDataOutputSchema, required=True)
    woody_biomass = fields.List(fields.List(fields.Float), required=True)
    balance = fields.Nested(MassBalanceData, required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeModel(**data)


def create(
    tree_params,
    tree_growth,
    pool_params,
    no_of_years,
    yearPlanted=0,
    initialStandDens=0,  # Should initialStandDens be 200?
    thin=None,
    mort=None,
) -> TreeModel:
    """Intialise TreeModel object (run biomass model, essentially).

    Args:
        tree_params TreeParams object (holds tree params)
        growth: tree_growth.Growth object
        pool_params: dict with alloc, turnover, thinFrac, and mortFrac
        yearPlanted: year (after start of project) tree is planted
        thin: thinning vector
        thinFrac: vector with the retained fraction of thinned biomass
                    for each pool (default = [leaf=1,branch=0,stem=0,
                                            croot=1,froot=1])
        mort: mortality vector
        mortFrac: vector with the retained fraction of dead biomass
                    for each pool (default same as thinFrac)
    Raises:
        KeyError: if pool_params doesn't have the appropriate keys

    """
    alloc = pool_params["alloc"]
    turnover = pool_params["turnover"]
    thin_frac = pool_params["thinFrac"]
    mort_frac = pool_params["mortFrac"]
    thin = np.zeros(no_of_years + 1) if thin is None else thin
    mort = np.zeros(no_of_years + 1) if mort is None else mort
    initial_biomass = tree_growth.fit_data[0]

    output, woody_biomass, balance = get_inputs(
        tree_params=tree_params,
        tree_growth=tree_growth,
        alloc=alloc,
        turnover=turnover,
        thin_frac=thin_frac,
        mort_frac=mort_frac,
        initial_biomass=initial_biomass,
        year_planted=yearPlanted,
        initial_stand_dens=initialStandDens,
        thin=thin,
        mort=mort,
        no_of_years=no_of_years,
    )

    params = {
        "tree_params": vars(tree_params),
        "tree_growth": vars(tree_growth),
        "alloc": alloc,
        "turnover": turnover,
        "thin_frac": thin_frac,
        "mort_frac": mort_frac,
        "thin": thin,
        "mort": mort,
        "output": output,
        "woody_biomass": woody_biomass,
        "balance": balance,
    }

    schema = TreeModelSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree model: {errors}")

    return schema.load(params)  # type: ignore


def from_defaults(
    tree_params,
    tree_growth,
    no_of_years,
    yearPlanted=0,
    standard_density=100,  # Should standard_density be 200?
    thin=None,
    thinFrac=None,
    mort=None,
    mortFrac=None,
):
    """Use defaults for pool params.
    Can override defaults for thinFrac and mortFrac by providing arguments.

    """

    data = csv_handler.read_csv("biomass_pool_params.csv", cols=(3, 4, 5, 6))
    turnover = data[:, 0]
    alloc = data[:, 1]
    thinFrac_temp = data[:, 2]
    mortFrac_temp = data[:, 3]

    # Take into account croot alloc - rs * stem alloc
    alloc[3] = alloc[2] * tree_params.root_to_shoot

    # thinning and mortality
    if thin is None:
        thin = np.zeros(no_of_years + 1)
    if thinFrac is None:
        thinFrac = thinFrac_temp
    if mort is None:
        mort = np.zeros(no_of_years + 1)
    if mortFrac is None:
        mortFrac = mortFrac_temp

    params = {
        "alloc": alloc,
        "turnover": turnover,
        "thinFrac": thinFrac,
        "mortFrac": mortFrac,
    }
    return create(
        tree_params=tree_params,
        tree_growth=tree_growth,
        pool_params=params,
        no_of_years=no_of_years,
        yearPlanted=yearPlanted,
        initialStandDens=standard_density,
        thin=thin,
        mort=mort,
    )


def get_inputs(
    thin,
    mort,
    turnover,
    thin_frac,
    mort_frac,
    alloc,
    tree_growth,
    tree_params,
    initial_biomass,
    year_planted,
    initial_stand_dens,
    no_of_years,
):
    """
    Calculate and return residues and soil inputs from the tree.

    Args:
        same explanations as __init__
    Returns:
        output: dict with soil,fire inputs due to this tree
                        (keys='carbon','nitrogen','DMon','DMoff')
        woody_biomass: vector with yearly woody biomass pools

    **NOTE** a lot of these arrays are implicitly 2d with 2nd
    dimension = [leaf, branch, stem, croot, froot]. Careful here.

    """
    # NOTE - initially quantities are in kg C
    #   -> woody_biomass and output get converted at end before returning

    # First get params from bpFile and ppFile
    print("new tree cohort running...")
    print(initial_biomass, year_planted, initial_stand_dens)

    yp = year_planted
    print("yp,  then initialSD")
    print(yp)
    print(initial_stand_dens)

    standard_density = np.zeros(no_of_years + 1)
    standard_density[yp] = initial_stand_dens
    print("standard_density:")
    print(standard_density)

    inputParams = {
        "live": np.array((no_of_years + 1) * [turnover]),
        "thin": thin,
        "dead": mort,
    }
    retainedFrac = {"live": 1, "thin": thin_frac, "dead": mort_frac}

    # initialise stuff
    pools = np.zeros((no_of_years + 1, 5))
    woody_biomass = np.zeros((no_of_years + 1, 5))
    tNPP = np.zeros(no_of_years + 1)

    flux = {}
    inputs = {}
    exports = {}
    for s in ["live", "dead", "thin"]:
        flux[s] = np.zeros((no_of_years + 1, 5))
        inputs[s] = np.zeros((no_of_years + 1, 5))
        exports[s] = np.zeros((no_of_years + 1, 5))

    inputC = np.zeros((no_of_years + 1, 5))
    exportC = np.zeros((no_of_years + 1, 5))
    biomGrowth = np.zeros((no_of_years + 1, 5))

    # set woody_biomass[0] to initial (allocated appropriately)
    pools[yp] = initial_biomass * alloc
    woody_biomass[yp] = pools[yp] * standard_density[yp]
    for s in inputs:
        flux[s][yp] = woody_biomass[yp] * inputParams[s][yp]

    in_ = np.zeros(no_of_years + 1)
    acc = np.zeros(no_of_years + 1)
    bal = np.zeros(no_of_years + 1)
    out = np.zeros(no_of_years + 1)

    for i in range(1 + year_planted, no_of_years + 1):
        # Careful with indices - using 1-based here
        #   since, e.g., woody_biomass[2] should correspond to
        #   biomass after 2 years

        agb = pools[i - 1][1] + pools[i - 1][2]

        # Growth for one tree
        tNPP[i] = derivative_functions[tree_growth.best](tree_growth.fit_params, agb)
        biomGrowth[i] = tNPP[i] * alloc * standard_density[i - 1]

        for s in inputs:
            flux[s][i] = inputParams[s][i] * pools[i - 1] * standard_density[i - 1]
            inputs[s][i] = retainedFrac[s] * flux[s][i]
            exports[s][i] = (1 - retainedFrac[s]) * flux[s][i]

        # Totals (in t C / ha)
        inputC[i] = sum(inputs.values())[i]
        exportC[i] = sum(exports.values())[i]

        woody_biomass[i] = woody_biomass[i - 1]
        woody_biomass[i] += biomGrowth[i]
        woody_biomass[i] -= sum(flux.values())[i]

        standard_density[i] = standard_density[i - 1]
        standard_density[i] *= 1 - (inputParams["dead"][i] + inputParams["thin"][i])
        if standard_density[i] < 1:
            print("SD [i] is less than 1, end of this tree cohort...")
            break
        pools[i] = woody_biomass[i] / standard_density[i]

        # Balance stuff

        in_[i] = biomGrowth[i].sum()
        acc[i] = woody_biomass[i].sum() - woody_biomass[i - 1].sum()
        out[i] = inputC[i].sum() + exportC[i].sum()
        bal[i] = in_[i] - out[i] - acc[i]

    # *********************
    # Standard output stuff
    # in tonnes
    # *********************
    massBalance = {
        "in_": in_ * 0.001,
        "out": out * 0.001,
        "acc": acc * 0.001,
        "bal": bal * 0.001,
    }
    woody_biomass *= 0.001  # convert to tonnes for emissions calc.
    C = inputC[0:no_of_years]
    DM = C / tree_params.carbon
    N = np.zeros((no_of_years, 5))
    for i in range(no_of_years):
        N[i] = tree_params.nitrogen[0:no_of_years] * DM[i]

    output = {}
    output["above"] = {
        "carbon": 0.001 * (C[:, 0] + C[:, 1] + C[:, 2]),
        "nitrogen": 0.001 * (N[:, 0] + N[:, 1] + N[:, 2]),
        "DMon": 0.001 * (DM[:, 0] + DM[:, 1] + DM[:, 2]),
        "DMoff": np.zeros(len(C[:, 0])),
    }
    output["below"] = {
        "carbon": 0.001 * ROOT_IN_TOP_30 * (C[:, 3] + C[:, 4]),
        "nitrogen": 0.001 * ROOT_IN_TOP_30 * (N[:, 3] + N[:, 4]),
        "DMon": 0.001 * ROOT_IN_TOP_30 * (DM[:, 3] + DM[:, 4]),
        "DMoff": np.zeros(len(C[:, 0])),
    }
    return output, woody_biomass, massBalance


def plot_biomass(tree_model, saveName=None):
    """Plot the biomass pool data."""

    fig = plt.figure()
    fig.suptitle("Biomass Pools")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tree_model.woody_biomass[:, 0], label="leaf")
    ax.plot(tree_model.woody_biomass[:, 1], label="branch")
    ax.plot(tree_model.woody_biomass[:, 2], label="stem")
    ax.plot(tree_model.woody_biomass[:, 3], label="coarse root")
    ax.plot(tree_model.woody_biomass[:, 4], label="fine root")
    ax.legend(loc="best")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Pool biomass (t C ha^-1)")
    ax.set_title("Biomass pools vs time")

    if saveName is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

    plt.close()


def plot_balance(tree_model, saveName=None):
    """Plot the mass balance data."""

    fig = plt.figure()
    fig.suptitle("Biomass Mass Balance")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tree_model.balance["in_"], label="in")
    ax.plot(tree_model.balance["out"], label="out")
    ax.plot(tree_model.balance["acc"], label="accumulated")
    ax.plot(tree_model.balance["bal"], label="balance")
    ax.legend(loc="best")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Biomass (t C ha^-1)")
    ax.set_title("Mass balance vs time")

    if saveName is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

    plt.close()


def print_biomass(tree_model):
    totalBiomass = np.sum(tree_model.woody_biomass, axis=1)

    # Prepare the data for tabulate
    table_data = [
        [year, f"{biomass:.2f}"]  # Format biomass to 2 decimal places
        for year, biomass in enumerate(totalBiomass)
    ]

    # Define headers
    headers = ["Year", "Biomass"]
    table_title = "BIOMASS MODEL"

    # Print the table using tabulate
    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(
        tabulate(table_data, headers=headers, numalign="center", tablefmt="fancy_grid")
    )


def print_balance(tree_model):
    print("\nMass-balance sum (kg C /ha): ", np.sum(tree_model.balance["bal"]))
    totDiff = np.sum(tree_model.balance["bal"]) / np.sum(tree_model.woody_biomass[-1])
    print("Normalized mass balance (kg C /ha): ", totDiff)


def save(tree_model, file="tree_model.csv"):
    """Save output and biomass to a csv file.
    Default path is in OUTPUT_DIR.

    Args:
        file: name or path to csv

    """
    # outputs
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(tree_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)

    # biomass
    biomass_file = file.split(".csv")[0] + "_biomass.csv"
    cols = ["leaf", "branch", "stem", "croot", "froot"]
    csv_handler.print_csv(
        biomass_file,
        tree_model.woody_biomass,
        col_names=cols,
        print_total=True,
        print_years=True,
    )


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
        from_defaults(
            tree_params=tree_params[i],
            tree_growth=growths[i],
            yearPlanted=int(csv_input_data[f"proj_plant_yr{i + 1}"]),
            standard_density=int(csv_input_data[f"proj_plant_dens{i + 1}"]),
            thin=thinning_project,
            thinFrac=thinning_fraction_left_project,
            mort=mortality_project,
            mortFrac=mortality_fraction_left_project,
            no_of_years=no_of_years,
        )
        for i in range(tree_count)
    ]
