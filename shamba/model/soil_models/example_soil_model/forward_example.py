from ..soil_model_types import ForwardSoilModelData, ForwardSoilModelBaseSchema


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
    solve_to_value=False,
) -> ForwardSoilModelData:
    schema = ForwardSoilModelBaseSchema()
    params = {}
    # This will fail because params is empty
    return schema.load(params)  # type: ignore
