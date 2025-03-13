#!/usr/bin/python
"""
Module containing tree growth information and allometric functions

class Growth:   for tree growth data and fitting
def ryan      allometric function based on C. Ryan biotropica paper (2010)

"""

import logging as log
import math
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marshmallow import Schema, fields, post_load
from scipy import optimize
from tabulate import tabulate

from .. import configuration
from ..common import csv_handler


# Functions to fit to
def hyperbolic_function(x, a, b):
    return a * (1 - np.exp(-b * x))


def exponential_function(x, a):
    return (1 + a) ** x - 1


def linear_function(x, a):
    return a * x


def logarithmic_function(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


fitting_functions = {
    "exp": exponential_function,
    "hyp": hyperbolic_function,
    "lin": linear_function,
    "log": logarithmic_function,
}


def exponential_function_derivative(fit_params, x):
    a = fit_params[0]
    return ((1 + a) ** x) * (np.log(1 + a))


def hyperbolic_function_derivative(fit_params, x):
    a = fit_params[0]
    b = fit_params[1]
    return a * b * np.exp(-b * x)


def linear_function_derivative(fit_params, lin_fn_inv, y):
    # TODO: this isn't being used and it doesn't do any side effects
    # x = lin_fn_inv(y)

    a = fit_params[0]
    return a


def logarithmic_function_inverse(fit_params, y):
    if math.fabs(y) < 0.00000001:
        x = 0
    else:
        a = fit_params[0]
        b = fit_params[1]
        c = fit_params[2]
        if y > a:  # should tend towards a
            x = a
        else:
            x = c + (math.log(y) - math.log(a - y)) / b

    return x


def logarithmic_function_derivative(fit_params, y):
    x = logarithmic_function_inverse(fit_params, y)

    a = fit_params[0]
    b = fit_params[1]
    c = fit_params[2]
    ret = (a * b * np.exp(-b * (x - c))) / ((np.exp(-b * (x - c)) + 1) ** 2)

    return ret


derivative_functions = {
    "exp": exponential_function_derivative,
    "hyp": hyperbolic_function_derivative,
    "lin": linear_function_derivative,
    "log": logarithmic_function_derivative,
}

"""
Object holding tree growth data, and fit functions/params for this.
Also includes method for printing and plotting data and fits

Instance variables
----------------
tree_params TreeParams object (holds carbon and dens, among others)
func        dict holding the four fitting functions
func_deriv   dict holding the four derivative functions
age         tree age data in years
diam        tree diameter data in cm
best        string with best fit ('exp, 'log', 'lin, or 'hyp')
fitDiam     dict holding fit diameter data in cm for all four fits
fit_params   dict holding fitting params for all four fits
fitMSE      dict holding MSE for all four fits

"""


class FitData(Schema):
    exp = fields.List(fields.Float(allow_nan=True), required=True)
    hyp = fields.List(fields.Float(allow_nan=True), required=True)
    lin = fields.List(fields.Float(allow_nan=True), required=True)
    log = fields.List(fields.Float(allow_nan=True), required=True)


class FitMSEData(Schema):
    exp = fields.Float(allow_nan=True)
    hyp = fields.Float(allow_nan=True)
    lin = fields.Float(allow_nan=True)
    log = fields.Float(allow_nan=True)


class TreeGrowth:
    def __init__(
        self,
        age,
        all_fit_data,
        all_fit_params,
        all_mse,
        allometric_key,
        best,
        biomass,
        fit_data,
        fit_mse,
        fit_params,
        tree_diameter,
    ):
        self.age = age
        self.all_fit_data = all_fit_data
        self.all_fit_params = all_fit_params
        self.all_mse = all_mse
        self.allometric_key = allometric_key
        self.best = best
        self.biomass = biomass
        self.fit_data = fit_data
        self.fit_mse = fit_mse
        self.fit_params = fit_params
        self.tree_diameter = tree_diameter


class TreeGrowthSchema(Schema):
    age = fields.List(fields.Float(), required=True)
    all_fit_data = fields.Nested(FitData, required=True)
    all_fit_params = fields.Nested(FitData, required=True)
    all_mse = fields.Nested(FitMSEData, required=True)
    allometric_key = fields.String(required=True)
    best = fields.String(required=True)
    biomass = fields.List(fields.Float(), required=True)
    fit_data = fields.List(fields.Float(), required=True)
    fit_mse = fields.Float(allow_nan=True)
    fit_params = fields.List(fields.Float(), required=True)
    tree_diameter = fields.List(fields.Float(), required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeGrowth(**data)


def get_biomass(tree_diameter, allometric_key, tree_params):
    # TODO: double check! modified code
    allometric_function = allometric[allometric_key]

    return np.array(
        [allometric_function(diameter, tree_params) for diameter in tree_diameter]
    )


def create(tree_params, growth_params, allom="chave dry") -> TreeGrowth:
    """Initialise tree growth data.

    Args:
        growth_params: dict with growth params
                        keys=are 'age','diam' OR 'age','biomass'
        tree_params: tree_params.TreeParams object (holds tree params)
        allom: which allometric to use - ignored if biomass is a key
                in growth_params since it won't be used
    Raises:
        KeyError: if dict doesn't have the right keys
        KeyError: if allom isn't in the allometric dict

    """
    tree_diameter = growth_params["diam"]
    allometric_key = allom.lower()
    biomass = (
        growth_params["biomass"]
        if "biomass" in growth_params
        else get_biomass(tree_diameter, allometric_key, tree_params)
    )
    age = growth_params["age"]

    all_fit_data, all_fit_params, all_mse = fit(age, biomass)

    # Find key with best fit
    best = min(all_mse, key=lambda k: all_mse[k])
    fit_data = all_fit_data[best]
    fit_params = all_fit_params[best]
    fit_mse = all_mse[best]

    params = {
        "age": age,
        "all_fit_data": all_fit_data,
        "all_fit_params": all_fit_params,
        "all_mse": all_mse,
        "allometric_key": allometric_key,
        "best": best,
        "biomass": biomass,
        "fit_data": fit_data,
        "fit_mse": fit_mse,
        "fit_params": fit_params,
        "tree_diameter": tree_diameter,
    }

    schema = TreeGrowthSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree growth: {errors}")

    return schema.load(params)


def from_csv1(
    tree_params,
    n,
    allometric_key,
    filename="growth_measurements.csv",
    isBiomass=False,
):
    """Construct Growth object using data in a csv file.

    When using intpu_csv from command line, constructing dictionary of
    np.arrays for each new cohort

    Args:
        tree: Tree object with tree information
        filename: csv to read data from with columns age,diam
        True if data in column 2 is biomass (not dbh)
        allom: which allometric to use
    Returns:
        Growth object
    Raises:
        IndexError: if file isn't the right format

    """

    data = pd.read_csv(configuration.INPUT_DIR + "/" + filename, sep=",")
    reader = data.loc[n]
    dictionary = reader.to_dict()

    age_input = ["age1", "age2", "age3", "age4", "age5", "age6"]
    age = {key: dictionary[key] for key in age_input}
    age = np.array(list(age.values())).astype(
        float
    )  # https://stackoverflow.com/questions/45957968/float-arguments-and-dict-values-with-numpy
    age = np.array(sorted(age, key=int))

    diam_input = ["diam1", "diam2", "diam3", "diam4", "diam5", "diam6"]
    diam = {key: dictionary[key] for key in diam_input}
    diam = np.array(list(diam.values())).astype(float)

    params = {
        "age": age,
        "diam": diam,
    }

    try:
        params = {
            "age": age,
            "diam": diam,
        }

        growth = create(tree_params, params, allometric_key)

    except IndexError:
        log.exception("Can't read growth data from %s " % filename)
        sys.exit(1)

    return growth


def from_csv2(
    tree_params,
    n,
    allometric_key,
    filename="growth_measurements.csv",
    isBiomass=False,
):
    """Construct Growth object using data in a csv file.

    When using intpu_csv from command line, constructing dictionary of
    np.arrays for each new cohort

    Args:
        tree: Tree object with tree information
        filename: csv to read data from with columns age,diam
        True if data in column 2 is biomass (not dbh)
        allom: which allometric to use
    Returns:
        Growth object
    Raises:
        IndexError: if file isn't the right format

    """

    data = pd.read_csv(configuration.INPUT_DIR + "/" + filename, sep=",")
    reader = data.loc[n]
    dictionary = reader.to_dict()

    age_input = [
        "sp2_age1",
        "sp2_age2",
        "sp2_age3",
        "sp2_age4",
        "sp2_age5",
        "sp2_age6",
    ]
    age = {key: dictionary[key] for key in age_input}
    age = np.array(list(age.values())).astype(float)
    age = np.array(sorted(age, key=int))

    diam_input = [
        "sp2_diam1",
        "sp2_diam2",
        "sp2_diam3",
        "sp2_diam4",
        "sp2_diam5",
        "sp2_diam6",
    ]
    diam = {key: dictionary[key] for key in diam_input}
    diam = np.array(list(diam.values())).astype(float)

    params = {
        "age": age,
        "diam": diam,
    }

    try:
        params = {
            "age": age,
            "diam": diam,
        }

        growth = create(tree_params, params, allometric_key)

    except IndexError:
        log.exception("Can't read growth data from %s " % filename)
        sys.exit(1)

    return growth


def from_csv3(
    tree_params,
    n,
    allometric_key,
    filename="growth_measurements.csv",
    isBiomass=False,
):
    """Construct Growth object using data in a csv file.

    When using intpu_csv from command line, constructing dictionary of
    np.arrays for each new cohort

    Args:
        tree: Tree object with tree information
        filename: csv to read data from with columns age,diam
        True if data in column 2 is biomass (not dbh)
        allom: which allometric to use
    Returns:
        Growth object
    Raises:
        IndexError: if file isn't the right format

    """

    data = pd.read_csv(configuration.INPUT_DIR + "/" + filename, sep=",")
    reader = data.loc[n]
    dictionary = reader.to_dict()

    age_input = [
        "sp3_age1",
        "sp3_age2",
        "sp3_age3",
        "sp3_age4",
        "sp3_age5",
        "sp3_age6",
    ]
    age = {key: dictionary[key] for key in age_input}
    age = np.array(list(age.values())).astype(float)
    age = np.array(sorted(age, key=int))

    diam_input = [
        "sp3_diam1",
        "sp3_diam2",
        "sp3_diam3",
        "sp3_diam4",
        "sp3_diam5",
        "sp3_diam6",
    ]
    diam = {key: dictionary[key] for key in diam_input}
    diam = np.array(list(diam.values())).astype(float)

    params = {
        "age": age,
        "diam": diam,
    }

    try:
        params = {
            "age": age,
            "diam": diam,
        }

        growth = create(tree_params, params, allometric_key)

    except IndexError:
        log.exception("Can't read growth data from %s " % filename)
        sys.exit(1)

    return growth


def from_arrays(tree, age, data, isBiomass=False, allometric_key="chave dry"):
    """Construct Growth object using data from numpy arrays."""
    pass


def fit(
    age, biomass
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Fit tree growth data using four fitting functions.

    Returns:
        **all dicts with keys 'lin','log','hyp','exp'**
        data: data points from each of the fits
        params: fitting parameters
        mse: mean-square error
    Raises:
        RuntimeError: if curve_fit can't fit the data to a curve

    """

    data = {}
    params = {}
    mse = {}
    init = {"exp": [20], "hyp": [200, 0.1], "lin": [1], "log": [100, 0, 0]}

    # TODO: this isn't be used?
    # numParams = {"exp": 1, "hyp": 2, "lin": 1, "log": 3}

    for curve in fitting_functions:
        # Do the fitting
        try:
            fit_params = optimize.curve_fit(
                fitting_functions[curve], age, biomass, init[curve]
            )
            par = fit_params[0]
            params[curve] = par

        except RuntimeError:
            log.warning("Could not fit data to %s", curve)
            fit_params = None
            par = None

        # Find data corresponding to those fitting params
        # Set data and params to array of nan if there's no fit
        if curve == "log":
            if fit_params is not None:
                data[curve] = fitting_functions[curve](age, par[0], par[1], par[2])
            else:
                params[curve] = np.array(3 * [np.nan])
                data[curve] = np.array(len(age) * [np.nan])

        elif curve == "lin" or curve == "exp":
            if fit_params is not None:
                data[curve] = fitting_functions[curve](age, par[0])
            else:
                params[curve] = np.array(1 * [np.nan])
                data[curve] = np.array(len(age) * [np.nan])
        else:  # hyp
            if fit_params is not None:
                data[curve] = fitting_functions[curve](age, par[0], par[1])
            else:
                params[curve] = np.array(2 * [np.nan])
                data[curve] = np.array(len(age) * [np.nan])

        # Find mse
        # set to inf if there's no fit for a given curve
        if fit_params is not None:
            mse[curve] = mse_fn(biomass, data[curve])
        else:
            mse[curve] = np.inf

    return data, params, mse


# TODO: this is not being used?
# def hyp_fn_inv(fit_params, y):
#     if math.fabs(y) < 0.00000001:
#         x = 0
#     else:
#         a = fit_params[0]
#         b = fitParmas[1]
#         if y > a:
#             x = a
#         else:
#             x = (math.log(a) - math.log(a - y)) / b

#     return x

# TODO: this is not being used?
# def lin_fn_inv(self, y):
#     if math.fabs(y) < 0.00000001:
#         x = 0
#     else:
#         a = float(self.fit_params[0])  # just to be safe
#         x = y / a

#     return x

# TODO: this is not being used?
# def exp_fn_inv(self, y):
#     if math.fabs(y) < 0.00000001:
#         x = 0
#     else:
#         a = self.fit_params[0]
#         x = math.log(y + 1) / math.log(a + 1)

#     return x


def mse_fn(yMean, yReal):
    return ((yMean - yReal) ** 2).mean()


def plot(tree_growth, fit=True, saveName=None):
    """Plot growth data and all four fits in a matplotlib figure."""

    fig = plt.figure()
    fig.suptitle("Tree Growth")

    ax = fig.add_subplot(1, 1, 1)

    ax.plot(tree_growth.age, tree_growth.biomass, "o", label="data")

    if fit:
        for curve in tree_growth.all_fit_data:
            ax.plot(
                tree_growth.age,
                tree_growth.all_fit_data[curve],
                "-",
                label="%s fit" % curve,
            )

        ax.legend(loc="best")

    ax.set_title("Tree biomass vs. Age")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Biomass (kg C /ha) ")

    # Shift ticks so points not cut off
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xmin = xticks[0] - 0.5 * (xticks[1] - xticks[0])
    xmax = xticks[-1] + 0.5 * (xticks[-1] - xticks[-2])
    ymin = yticks[0] - 0.5 * (yticks[1] - yticks[0])
    ymax = yticks[-1] + 0.5 * (yticks[-1] - yticks[-2])
    ax.set_xlim(xmin, xmax)

    if saveName is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))


def print_to_stdout(tree_growth, label, fit=True, params=True, mse=True):
    """Print data and fits for tree growth data to stdout using tabulate."""
    # Prepare the data for tabulate
    table_data = [
        [age, f"{diameter:.2f}", f"{biomass:.2f}"]
        for age, diameter, biomass in zip(
            tree_growth.age, tree_growth.tree_diameter, tree_growth.biomass
        )
    ]

    # Define headers
    headers = ["Age (years)", "Diameter (cm)", "Biomass (kg C)"]

    table_title = f"TREE GROWTH Data for {label}"

    # Print the table using tabulate
    print()  # Newline
    print()  # Newline
    print(table_title)
    print(f"Allometric: {tree_growth.allometric_key}")
    print("=" * len(table_title))
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    if fit:
        table_data = [
            [f"{data:.2f}", f"{exp:.2f}", f"{hyp:.2f}", f"{lin:.2f}", f"{log:.2f}"]
            for data, exp, hyp, lin, log in zip(
                tree_growth.biomass,
                tree_growth.all_fit_data["exp"],
                tree_growth.all_fit_data["hyp"],
                tree_growth.all_fit_data["lin"],
                tree_growth.all_fit_data["log"],
            )
        ]
        headers = ["Data", "Exp.", "Hyp.", "Lin.", "Log."]

        print()  # Newline
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    if params and mse:
        table_data = [
            [
                "MSE",
                f"{tree_growth.all_mse['exp']:.2f}",
                f"{tree_growth.all_mse['hyp']:.2f}",
                f"{tree_growth.all_mse['lin']:.2f}",
                f"{tree_growth.all_mse['log']:.2f}",
            ],
            [
                "a",
                f"{tree_growth.all_fit_params['exp'][0]:.2f}",
                f"{tree_growth.all_fit_params['hyp'][0]:.2f}",
                f"{tree_growth.all_fit_params['lin'][0]:.2f}",
                f"{tree_growth.all_fit_params['log'][0]:.2f}",
            ],
            [
                "b",
                "",
                f"{tree_growth.all_fit_params['hyp'][1]:.2f}",
                "",
                f"{tree_growth.all_fit_params['log'][1]:.2f}",
            ],
            ["c", "", "", "", f"{tree_growth.all_fit_params['log'][2]:.2f}"],
        ]
        headers = ["", "Exp.", "Hyp.", "Lin.", "Log."]

        print()  # Newline
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


def save(tree_growth, file="tree_growth.csv"):
    """Save growth stuff to a csv file
    Default path is in OUTPUT_DIR.

    Args:
        file: name or path to csv file

    """
    # growth data
    csv_handler.print_csv(
        file,
        np.column_stack(
            (tree_growth.age, tree_growth.tree_diameter, tree_growth.biomass)
        ),
        col_names=["age", "diam", "biomass", "allom=" + tree_growth.allometric_key],
    )

    # fitted data - in a list because of the size heterogeneity
    fit_file = file.split(".csv")[0] + "_fit.csv"
    data = np.column_stack(
        (
            tree_growth.biomass,
            tree_growth.all_fit_data["exp"],
            tree_growth.all_fit_data["hyp"],
            tree_growth.all_fit_data["lin"],
            tree_growth.all_fit_data["log"],
        )
    )
    cols = ["data", "exp", "hyp", "lin", "log"]
    csv_handler.print_csv(fit_file, data, col_names=cols)

    # fit parameters
    param_file = file.split(".csv")[0] + "_fit_params.csv"
    temp = {"exp": [], "hyp": [], "lin": [], "log": []}
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for s in ["exp", "hyp", "lin", "log"]:
        row1.append(tree_growth.all_mse[s])
        row2.append(tree_growth.all_fit_params[s][0])
        try:
            row3.append(tree_growth.all_fit_params[s][1])
        except IndexError:
            row3.append("")
        try:
            row4.append(tree_growth.all_fit_params[s][2])
        except IndexError:
            row4.append("")

    data = [row1, row2, row3, row4]
    cols = ["exp", "hyp", "lin", "log"]
    csv_handler.print_csv(param_file, data, col_names=cols)


# --------------------------------------------------
# Begin methods for allometric equations
# NOTE: RETURNS kg C - WHEN USING WITH CARBON POOLS
# (TREE MODEL), MAKE SURE TO CONVERT TO t C
# --------------------------------------------------

# Return AGB of general allometric of type diam = -c*(agb)^c
#   -> solving for agb gives agb = exp[(ln(diam) - ln(c)) / d]
#   -> agb = exp[a + b*ln(diam)]    since c,d are arbitrary


def log_allom(params, d, dens=None):
    """General log type allometric.

    Args:
        params: array-like holding fit params
                with p[0] multiplied by highest power of ln(d)
                e.g. p=np.array([2.601, -3.629])
                     -> agb = exp(2.601*log(d)-3.629) (kg) is the ryan allom
        d: diameter
        dens: tree density (only needed for some allometrics)
    Returns:
        agb: agb corresponding to diameter of d

    """
    if math.fabs(d) < 0.00000001:
        agb = 0
    else:
        logd = math.log(d)
        agb = np.polyval(params, logd)
        agb = math.exp(agb)  # kg C
        if dens is not None:  # for Chave allometric (and possibly others)
            agb *= dens

    return agb


# Some specific log allometrics
# All take dbh vectors and tree object as arguments
def ryan(dbh, tree_params):
    """C. Ryan, biotropica (2010)."""
    return log_allom([2.601, -3.629], dbh)


def tumwebaze_grevillea(dbh, tree_params):
    """Tumwebaze et al. (2013) - Grevillea."""

    agb = log_allom([3.06, -5.5], dbh) + log_allom([1.32, 1.06], dbh)
    return agb * tree_params.carbon


def tumwebaze_maesopsis(dbh, tree_params):
    """Tumwebaze et al. (2013) - Maesopsis."""

    agb = log_allom([3.33, -7.02], dbh) + log_allom([2.38, -2.9], dbh)
    return agb * tree_params.carbon


def tumwebaze_markhamia(dbh, tree_params):
    """Tumwebaze et al. (2013) - Markhamia."""
    agb = log_allom([2.63, -4.91], dbh) + log_allom([2.43, -3.08], dbh)
    return agb * tree_params.carbon


def chave_dry(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with < 1500 mm/year rainfall, > 5 months dry season

    """
    agb = log_allom([-0.0281, 0.207, 1.784, -0.667], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon


def chave_moist(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with 1500-3000 mm/year rainfall, 1-4 months dry season

    """
    agb = log_allom([-0.0281, 0.207, 2.148, -1.499], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon


def chave_wet(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with > 3500 mm/year rainfall, no seasonality

    """
    agb = log_allom([-0.0281, 0.207, 1.98, -1.239], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon


allometric = {
    "ryan": ryan,
    "grevillea": tumwebaze_grevillea,
    "maesopsis": tumwebaze_maesopsis,
    "markhamia": tumwebaze_markhamia,
    "chave dry": chave_dry,
    "chave moist": chave_moist,
    "chave wet": chave_wet,
    "log_allom": log_allom,
}
