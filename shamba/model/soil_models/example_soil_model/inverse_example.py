import numpy as np

from ..soil_model_types import InverseSoilModelData, InverseSoilModelBaseSchema

def create(soil, climate, cover=np.ones(12)) -> InverseSoilModelData:
    schema = InverseSoilModelBaseSchema()
    params = {}
    # This will fail because params is empty
    return schema.load(params)  # type: ignore
