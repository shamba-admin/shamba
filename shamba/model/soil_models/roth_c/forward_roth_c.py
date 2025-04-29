import math
import os

import matplotlib.pyplot as plt
import numpy as np
from marshmallow import fields, post_load
from scipy import integrate
from tabulate import tabulate
from typing import List, Tuple, Optional

from ... import configuration, emit
from ...common import csv_handler
from .roth_c import RothCSchema
from .roth_c import create as create_roth_c
from .roth_c import dC_dt


class ForwardRothCData:
    """
    Forward RothC object

    Instance variables
    ----------------
    SOC            vector with soil distributions for each year
    soil_params    SoilParams object with soil params (porosity, field capacity, etc.)
    climate        Climate object with climate data (rain, evaporation, etc.)
    cover          vector with cover for each year
    k              vector with crop coefficient for each year
    inputs         vector with inputs for each year
    Cy0Year        vector with initial soil carbon for each year

    """

    def __init__(
        self,
        soil_params,
        climate,
        cover,
        k,
        SOC,
        inputs,
        Cy0Year,
    ):
        self.soil = soil_params
        self.climate = climate
        self.cover = cover
        self.k = k
        self.SOC = SOC
        self.inputs = inputs
        self.Cy0Year = Cy0Year


class ForwardRothCSchema(RothCSchema):
    SOC = fields.List(fields.List(fields.Float), required=True)
    inputs = fields.List(fields.List(fields.Float), required=True)
    Cy0Year = fields.Float(required=True)

    @post_load
    def build_forward_roth_c(self, data, **kwargs):
        roth_c_data = {k: data[k] for k in RothCSchema().fields.keys()}
        forward_data = {k: data[k] for k in ["SOC", "inputs", "Cy0Year"]}
        return ForwardRothCData(**forward_data, **roth_c_data)


def create(
    soil,
    climate,
    cover,
    Ci,
    no_of_years,
    crop=[],
    tree=[],
    litter=[],
    fire=[],
    solveToValue=False,
) -> ForwardRothCData:
    """Creates ForwardRothCData.

    Args:
        soil: soil object
        climate: climate object
        cover: soil cover vector
        Ci: initial carbon pools (for solver)
        crop: list of crop objects which provide carbon to soil
        tree: list of tree objects which provide carbon to soil
        litter: list of litter objects which provide carbon to soil
        solveToValue: whether to solve to value (to Cy0) or by time

    """
    roth_c = create_roth_c(soil, climate, cover)

    SOC, inputs, Cy0Year = solver(
        roth_c, Ci, no_of_years, crop, tree, litter, fire, solveToValue
    )

    params = {
        "soil_params": vars(roth_c.soil),
        "climate": vars(roth_c.climate),
        "cover": roth_c.cover,
        "k": roth_c.k,
        "SOC": SOC,
        "inputs": inputs,
        "Cy0Year": Cy0Year,
    }

    schema = ForwardRothCSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in ForwardRothC data: {str(errors)}")

    return schema.load(params)  # type: ignore


def solver(
    roth_c, Ci, no_of_years, crop=[], tree=[], litter=[], fire=[], solveToValue=False
):
    """Run RothC in 'forward' mode;
    solve dC_dt over a given time period
    or to a certain value, given a vector with soil inputs.
    Use scipy.integrate.odeint as a Runge-Kutta solver.

    Args:
        crop: list of Crop objects (not reduced by fire)
        tree: list of Tree objects (not reduced by fire)
        solveToValue: whether to solve to a particular value (Cy0)
                        as opposed to for a certain amount of time
    Returns:
        C: vector with yearly distribution of soil carbon
        inputs: yearly inputs to soil with 2 columns (crop,tree)
        year_target_reached: year that the target value
                (if solveToValue==True) of Cy0 was reached
                since it will (probably) not be on nice round num.

    """

    # Reduce inputs due to fire
    soilIn_crop, soilIn_tree = emit.reduce_from_fire(
        no_of_years=no_of_years, crop=crop, tree=tree, litter=litter, fire=fire
    )

    # make input into array with 2 columns (soilIn_crop,soilIn_tree)
    inputs = np.column_stack((soilIn_crop, soilIn_tree))

    # Calculate yearly values of x based dpm:rpm ratio for each year
    x = get_partitions(roth_c, inputs, no_of_years)
    t = np.arange(0, 1, 0.001)
    C = np.zeros((no_of_years + 1, 4))
    C[0] = Ci
    Ctot = np.zeros(no_of_years + 1)

    # keep track of when target year reached
    year_target_reached = 0
    # To keep track of closest value to SOCf
    if solveToValue:
        prevDiff_outer = 1000.0
        prevDiff_inner = 1000.0

    for i in range(1, no_of_years + 1):
        # Careful with indices - using 1-based for arrays here
        # since, e.g., C[2] should correspond to carbon after 2 years

        # Solve the diffEQs to get pools for year i
        Ctemp = integrate.odeint(
            dC_dt, C[i - 1], t, args=(x[i - 1], roth_c.k, inputs[i - 1].sum())
        )
        C[i] = Ctemp[-1]  # carbon pools at end of year

        # Check to see if close to target value
        if solveToValue:
            j = -1
            for c in Ctemp:
                j += 1
                Ctot = c.sum() + roth_c.soil.iom
                currDiff = math.fabs(Ctot - roth_c.soil.Cy0)

                if currDiff - prevDiff_inner < 0.00000001:
                    prevDiff_inner = currDiff
                    c_inner = c
                    closest_j = j
                else:  # getting farther away
                    break

            if prevDiff_inner < prevDiff_outer:
                prevDiff_outer = prevDiff_inner
                c_outer = c_inner
                closest_i = i
            else:
                break

    if solveToValue:
        C[closest_i] = c_outer
        C = C[0 : closest_i + 1]
        year_target_reached = float(closest_j) / len(t)
        inputs = inputs[0 : len(C[:, 0]) - 1]

    return C, inputs, year_target_reached


def get_partitions(roth_c, inputs, no_of_years):
    """Calculate partitioning coefficients.

    Args:
        input: 2d array with crop_in,tree_in inputs as the columns
    Returns:
        x: partitioning coefficient (as a vector for each year)

    """

    # Determine p_2 (fraction of input going to CO2) based on clay content
    z = 1.67 * (1.85 + 1.6 * math.exp(-0.0786 * roth_c.soil.clay))
    p2 = z / (z + 1)

    # p_3 is always 0.46 (see RothC papaer
    p3 = 0.46

    # p_1 is dpm fraction of input:
    #   deciduous tropical woodland: dpm=0.2, rpm=0.8
    #   crops: dpm=0.59, rpm=0.41
    # So weigh p_1 according to amount of trees/crops for each year

    # Normalize
    i = 0
    normInput = np.zeros((no_of_years, 2))
    for row in inputs:
        if math.fabs(float(row.sum())) < 0.00000001:
            normInput[i] = 0
        else:
            normInput[i] = inputs[i] / float(row.sum())
        i += 1

    # Weighted mean of dpm
    p1 = np.array(0.59 * normInput[:, 0] + 0.2 * normInput[:, 1])

    # Construct x
    # make arrays (p1 already array)
    x = np.column_stack(
        (
            p1,
            1 - p1,
            np.array(no_of_years * [p3 * (1 - p2)]),
            np.array(no_of_years * [(1 - p2) * (1 - p3)]),
        )
    )

    return x


def plot(forward_roth_c, legend_string, no_of_years, save_name=None):
    """Plot total carbon vs year for forwardRothC run.

    Args:
        legend_string: string to put in legend

    """
    fig = plt.figure()
    fig.suptitle("Soil Carbon")  # Replace set_window_title with suptitle
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("SOC (t C ha^-1)")
    ax.set_title("Total soil carbon vs time")

    tot_soc = np.sum(forward_roth_c.SOC, axis=1)
    if len(tot_soc) == no_of_years + 1:
        # baseline or project
        x = list(range(len(tot_soc)))
    else:
        # initialisation run is before year 0
        x = np.array(list(range(-len(tot_soc) + 2, 2)))
        x = x - forward_roth_c.Cy0Year
        x[-1] = 0

    tot_soc = np.sum(forward_roth_c.SOC, axis=1)
    ax.plot(x, tot_soc, label=legend_string)
    ax.legend(loc="best")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))


def create_row(
    year: float, carbon: float, inputs: Optional[List[float]] = None
) -> List[float]:
    return [year, carbon] + (inputs or [])


def generate_table_data(
    tot_soc: np.ndarray,
    soil_iom: float,
    inputs: List[Tuple[float, float]],
    years: np.ndarray,
) -> List[List[float]]:
    return [
        create_row(year, soc + soil_iom, list(inputs[i]) if i < len(inputs) else [])
        for i, (year, soc) in enumerate(zip(years, tot_soc))
    ]


def print_to_stdout(forward_roth_c, no_of_years: int, label: str) -> None:
    """Print data from forward RothC run to stdout using tabulate with a functional approach."""
    table_title = f"FORWARD CALCULATIONS for {label}"

    tot_soc = np.sum(forward_roth_c.SOC, axis=1)
    soil_iom = forward_roth_c.soil.iom

    if len(tot_soc) == no_of_years + 1:
        years = np.arange(len(tot_soc), dtype=float)
    else:
        years = np.arange(-len(tot_soc) + 2, 2, dtype=float) - forward_roth_c.Cy0Year
        years[-1] = 0

    table_data = generate_table_data(tot_soc, soil_iom, forward_roth_c.inputs, years)

    headers = ["Year", "Carbon", "Crop In", "Tree In"]

    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="fancy_grid"))


def save(forward_roth_c, no_of_years, file="soil_model_forward.csv"):
    """Save data from forward RothC run to a csv.
    Default path is OUTPUT_DIR.

    """
    tot_soc = np.sum(forward_roth_c.SOC, axis=1)
    inputs = np.append(forward_roth_c.inputs, [[0, 0]], axis=0)
    data = np.column_stack(
        (
            tot_soc + forward_roth_c.soil.iom,
            forward_roth_c.SOC,
            np.array(len(tot_soc) * [forward_roth_c.soil.iom]),
            inputs[:, 0],
            inputs[:, 1],
        )
    )
    cols = ["soc", "dpm", "rpm", "bio", "hum", "iom", "crop_in", "tree_in"]
    if len(tot_soc) != no_of_years + 1:  # solve to value
        cols.insert(0, "year")
        x = np.array(list(range(-len(tot_soc) + 2, 2)))
        x = x - forward_roth_c.Cy0Year
        x[-1] = 0
        data = np.column_stack((x, data))
        csv_handler.print_csv(file, data, col_names=cols)
    else:
        csv_handler.print_csv(file, data, col_names=cols, print_years=True)
