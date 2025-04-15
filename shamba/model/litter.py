import logging as log

import numpy as np
from marshmallow import Schema, fields, post_load

from .common import csv_handler
from .common_schema import OutputSchema as LitterDataOutputSchema


class LitterModelData:
    """
    Litter model object. Read litter params
    and calculate residues and inputs to the soil.

    Instance variables
    ------------------
    carbon      litter carbon content
    nitrogen    litter nitrogen content
    output      output to soil,fire (dict with keys 'DMon','DMoff,'carbon','nitrogen')
    """

    def __init__(self, carbon, nitrogen, output):
        self.carbon = carbon
        self.nitrogen = nitrogen
        self.output = output


class LitterDataSchema(Schema):
    carbon = fields.Float(required=True)
    nitrogen = fields.Float(required=True)
    output = fields.Nested(LitterDataOutputSchema)

    @post_load
    def build(self, data, **kwargs):
        return LitterModelData(**data)


def create(
    litter_params, litterFreq, litterQty, no_of_years, litterVector=None
) -> LitterModelData:
    """Create LitterModelData object.

    Args:
        litter_params:  dict with litter params(keys='carbon','nitrogen')
        litterFreq:     frequency of litter application
        litterQty:      Quantity (in t C ha^-1) of litter when applied.
        litterVector:   vector with custom litter additions (t DM / ha)
                        (e.g. for when litter isn't at regular freq.)
                        -> overrides any quantity and freq. info
    Returns:
        LitterModelData: object containing litter parameters
    """

    carbon = litter_params["carbon"]
    nitrogen = litter_params["nitrogen"]
    params = {
        "carbon": carbon,
        "nitrogen": nitrogen,
        "output": get_inputs(
            carbon=carbon,
            nitrogen=nitrogen,
            litterFreq=litterFreq,
            litterQty=litterQty,
            litterVector=litterVector,
            no_of_years=no_of_years,
        ),
    }

    schema = LitterDataSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in litter model data: {str(errors)}")

    return schema.load(params)  # type: ignore


def from_csv(
    litterFreq, litterQty, no_of_years, filename="litter.csv", row=0, litterVector=None
):
    """
    Same as create, but with litter parameters from a csv file.
    """

    data = csv_handler.read_csv(filename)
    data = np.atleast_2d(data)
    try:
        params = {"carbon": data[row, 0], "nitrogen": data[row, 1]}
        litter = create(
            litter_params=params,
            litterFreq=litterFreq,
            litterQty=litterQty,
            no_of_years=no_of_years,
            litterVector=litterVector,
        )
    except IndexError:
        log.exception("Can't find row %d in %s" % (row, filename))
        raise

    return litter


def from_defaults(litterFreq, litterQty, no_of_years, litterVector=None):
    """
    Same as create, but with default litter parameters.
    """

    # Carbon and nitrogen content of litter input defaults
    params = {"carbon": 0.5, "nitrogen": 0.018}
    return create(
        litter_params=params,
        litterFreq=litterFreq,
        litterQty=litterQty,
        no_of_years=no_of_years,
        litterVector=litterVector,
    )


def synthetic_fert(freq, qty, nitrogen, no_of_years, vector=None):
    """Synthetic fertiliser (special case of litter).
    Be sure to keep separate though when passing a litter object to
    other methods/classes. (e.g. fert isn't an input to soil model)"""
    params = {"carbon": 0, "nitrogen": nitrogen}

    return create(
        litter_params=params,
        litterFreq=freq,
        litterQty=qty,
        no_of_years=no_of_years,
        litterVector=vector,
    )


def get_inputs(carbon, nitrogen, litterFreq, litterQty, litterVector, no_of_years):
    """Calculate and return DM, C, and N inputs to
    soil from additional litter.

    Args:
        carbon: litter carbon content
        nitrogen: litter nitrogen content
        litterFreq: frequency of litter application
        litterQty: amount of dry matter added to field when litter added in t DM ha^-1
        litterVector: vector with custom litter additions (t DM / ha)
        no_of_years: number of years in the project
    Returns:
        output: dict with soil,fire inputs due to litter (keys='carbon','nitrogen','DMon','DMoff')

    """

    if litterVector is None:
        # Construct vectors for DM, C, N
        Cinput = np.zeros(no_of_years)
        Ninput = np.zeros(no_of_years)
        DMinput = np.zeros(no_of_years)

        # loop through years when litter is added
        if litterFreq == 0:
            years = 0
        else:
            years = list(range(-1, no_of_years, litterFreq))
            years = years[1:]
        DMinput[years] = litterQty
        Cinput[years] = litterQty * carbon
        Ninput[years] = litterQty * nitrogen
    else:
        # DM vector already specified
        DMinput = np.array(litterVector)
        Cinput = DMinput * carbon
        Ninput = DMinput * nitrogen

    # Standard output (same as crop and tree classes)
    output = {}
    output["above"] = {
        "carbon": Cinput,
        "nitrogen": Ninput,
        "DMon": DMinput,
        "DMoff": np.zeros(len(Cinput)),
    }
    output["below"] = {
        "carbon": np.zeros(len(Cinput)),
        "nitrogen": np.zeros(len(Cinput)),
        "DMon": np.zeros(len(Cinput)),
        "DMoff": np.zeros(len(Cinput)),
    }

    return output


def save(litter_model, file="litter_model.csv"):
    """Save output to a csv. Default path is OUTPUT_DIR

    Args:
        file: name or path to csv

    """
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(litter_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)
