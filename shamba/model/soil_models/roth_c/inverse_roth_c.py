import math

import numpy as np
from marshmallow import fields, post_load
from scipy import optimize

from ...common import csv_handler
from .roth_c import RothCData, RothCSchema
from .roth_c import create as create_roth_c
from .roth_c import dC_dt


class InverseRothCData(RothCData):
    """
    Inverse RothC model. Extends RothCData class.

    Instance variables
    ------------------
    eq_C     calculated equilibrium distribution of carbon
    input_C  yearly input to soil giving eq_C
    x       partitioning coefficients

    """

    def __init__(self, eq_C, input_C, x, **kwargs):
        super().__init__(**kwargs)
        self.eq_C = eq_C
        self.input_C = input_C
        self.x = x


class InverseRothCSchema(RothCSchema):
    eq_C = fields.List(fields.Float, required=True)
    input_C = fields.Float(required=True)
    x = fields.List(fields.Float, required=True)

    @post_load
    def build_inverse_roth_c(self, data, **kwargs):
        roth_c_data = {k: data[k] for k in RothCSchema().fields.keys()}
        inverse_data = {k: data[k] for k in ["eq_C", "input_C", "x"]}
        return InverseRothCData(**inverse_data, **roth_c_data)


def create(soil, climate, cover=np.ones(12)) -> InverseRothCData:
    """Creates inverse rothc object.

    Args:
        soil: soil object with soil parameters
        climate: climate object with climate parameters
        cover: monthly soil cover vector

    """
    roth_c = create_roth_c(soil, climate, cover)

    eq_C, input_C, x = solver(roth_c)

    params = {
        "soil_params": vars(roth_c.soil),
        "climate": vars(roth_c.climate),
        "cover": roth_c.cover,
        "k": roth_c.k,
        "eq_C": eq_C,
        "input_C": input_C,
        "x": x,
    }

    schema = InverseRothCSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in InverseRothC data: {str(errors)}")

    return schema.load(params)  # type: ignore


def solver(roth_c):
    """Run RothC in 'inverse' mode to find inputs
    giving equilibrium and Carbon pool values at equilibrium.

    Returns:
        eq_C: equilibrium distribution of carbon
        eq_input: yearly tree input giving equilibrium
        x: partitioning coefficient for pools

    """

    # Partioning coefficients
    x = get_partitions(roth_c)
    t = 0  # just needed to define for dC_dt function
    C0 = np.array([0.1, 10, 0, 0])  # initial ballpark guess

    # Keep track of difference b/t Ctot and Ceq
    previous_difference = 1000.0

    # Loop through range of inputs
    for input in np.arange(0.01, 10, 0.001):
        C = optimize.fsolve(dC_dt, C0, args=(t, x, roth_c.k, input))
        # TODO: Is this correct
        Ctot = np.sum(np.array(C)) + roth_c.soil.iom
        current_difference = math.fabs(Ctot - roth_c.soil.Ceq)

        if current_difference < previous_difference:
            previous_difference = current_difference
            input_C = input
            previous_C = C

        if previous_difference < current_difference:
            break

    eq_C = previous_C
    eq_input = input_C
    return eq_C, eq_input, x


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
    print("Equilibrium C -", inverse_roth_c.eq_C.sum() + inverse_roth_c.soil.iom)
    for i in range(len(inverse_roth_c.eq_C)):
        print("   ", pools[i], "- - - -", inverse_roth_c.eq_C[i])
    print("    IOM", "- - - -", inverse_roth_c.soil.iom)
    print("Equil. inputs -", inverse_roth_c.input_C)
    print("")


def save(inverse_roth_c, file="soil_model_inverse.csv"):
    """Save data to csv. Default path is OUTPUT_DIR."""

    data = np.array(
        [
            np.sum(inverse_roth_c.eq_C) + inverse_roth_c.soil.iom,
            inverse_roth_c.eq_C[0],
            inverse_roth_c.eq_C[1],
            inverse_roth_c.eq_C[2],
            inverse_roth_c.eq_C[3],
            inverse_roth_c.soil.iom,
            inverse_roth_c.input_C,
        ]
    )
    cols = ["Ceq", "dpm", "rpm", "bio", "hum", "iom", "inputs"]
    csv_handler.print_csv(file, data, col_names=cols)
