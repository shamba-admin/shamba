import math

import numpy as np
from marshmallow import fields, post_load
from scipy import optimize

from ....common import csv_handler
from .roth_c import RothCData, RothCSchema
from .roth_c import create as create_roth_c
from .roth_c import dC_dt

"""
Inverse RothC model. Extends RothC class.

Instance variables
------------------
eqC     calculated equilibrium distribution of carbon
inputC  yearly input to soil giving eqC
x       partitioning coefficients

"""


class InverseRothCData(RothCData):
    def __init__(self, eqC, inputC, x, **kwargs):
        super().__init__(**kwargs)
        self.eqC = eqC
        self.inputC = inputC
        self.x = x


class InverseRothCSchema(RothCSchema):
    eqC = fields.List(fields.Float, required=True)
    inputC = fields.Float(required=True)
    x = fields.List(fields.Float, required=True)

    @post_load
    def build_inverse_roth_c(self, data, **kwargs):
        roth_c_data = {k: data[k] for k in RothCSchema().fields.keys()}
        inverse_data = {k: data[k] for k in ["eqC", "inputC", "x"]}
        return InverseRothCData(**inverse_data, **roth_c_data)


def create(soil, climate, cover=np.ones(12)):
    """Initialise inverse rothc object.

    Args:
        soil: soil object with soil parameters
        climate: climate object with climate parameters
        cover: monthly soil cover vector

    """
    roth_c = create_roth_c(soil, climate, cover)

    eqC, inputC, x = solver(roth_c)

    params = {
        "soil_params": vars(roth_c.soil),
        "climate": vars(roth_c.climate),
        "cover": roth_c.cover,
        "k": roth_c.k,
        "eqC": eqC,
        "inputC": inputC,
        "x": x,
    }

    schema = InverseRothCSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in InverseRothC data: {str(errors)}")

    return schema.load(params)


def solver(roth_c):
    """Run RothC in 'inverse' mode to find inputs
    giving equilibrium and Carbon pool values at equilibrium.

    Returns:
        eqC: equilibrium distribution of carbon
        eqInput: yearly tree input giving equilibrium
        x: partitioning coefficient for pools

    """

    # Partioning coefficients
    x = get_partitions(roth_c)
    t = 0  # just needed to define for dC_dt function
    C0 = np.array([0.1, 10, 0, 0])  # initial ballpark guess

    # Keep track of difference b/t Ctot and Ceq
    prevDiff = 1000.0

    # Loop through range of inputs
    for input in np.arange(0.01, 10, 0.001):
        C = optimize.fsolve(dC_dt, C0, args=(t, x, roth_c.k, input))
        # TODO: Is this correct
        Ctot = np.sum(np.array(C)) + roth_c.soil.iom
        currDiff = math.fabs(Ctot - roth_c.soil.Ceq)

        if currDiff < prevDiff:
            prevDiff = currDiff
            inputC = input
            prevC = C
            prevCtot = Ctot

        if prevDiff < currDiff:
            break

    eqC = prevC
    eqInput = inputC
    return eqC, eqInput, x


def get_partitions(roth_c):
    """Calculate partitioning coefficients.

    Returns:
        x: partitioning coefficient for the 4 pools

    """

    # Determine p_2 (fraction of input going to CO2) based on clay content
    z = 1.67 * (1.85 + 1.6 * math.exp(-0.0786 * roth_c.soil.clay))
    p2 = z / (z + 1)

    # p_3 is always 0.46 (see RothC papaer
    p3 = 0.46

    # p_1 is dpm fraction of input:
    #   deciduous tropical woodland: dpm=0.2, rpm=0.8
    #   crops: dpm=0.59, rpm=0.41

    # no crops at equil so p1 always 0.2
    p1 = 0.2
    return np.array([p1, 1 - p1, p3 * (1 - p2), (1 - p2) * (1 - p3)])


def print_to_stdout(inverse_roth_c):
    """Print data from inverse RothC run to stdout."""

    pools = ["DPM", "RPM", "BIO", "HUM"]
    print("\nINVERSE CALCULATIONS")
    print("====================\n")
    print("Equilibrium C -", inverse_roth_c.eqC.sum() + inverse_roth_c.soil.iom)
    for i in range(len(inverse_roth_c.eqC)):
        print("   ", pools[i], "- - - -", inverse_roth_c.eqC[i])
    print("    IOM", "- - - -", inverse_roth_c.soil.iom)
    print("Equil. inputs -", inverse_roth_c.inputC)
    print("")


def save(inverse_roth_c, file="soil_model_inverse.csv"):
    """Save data to csv. Default path is OUTPUT_DIR."""

    data = np.array(
        [
            np.sum(inverse_roth_c.eqC) + inverse_roth_c.soil.iom,
            inverse_roth_c.eqC[0],
            inverse_roth_c.eqC[1],
            inverse_roth_c.eqC[2],
            inverse_roth_c.eqC[3],
            inverse_roth_c.soil.iom,
            inverse_roth_c.inputC,
        ]
    )
    cols = ["Ceq", "dpm", "rpm", "bio", "hum", "iom", "inputs"]
    csv_handler.print_csv(file, data, col_names=cols)
