from typing import List, Dict, Any, Optional
from toolz import get

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import handle_intervention
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
from model.soil_models.soil_model_types import SoilModelType

ForwardSoilModel = ForwardSoilModule.get_soil_model(SoilModelType.ROTH_C)
InverseSoilModel = InverseSoilModule.get_soil_model(SoilModelType.ROTH_C)

def run(project_name, data, use_api: bool):
    inputs: Optional[List[Dict[str, Any]]] = get("inputs", data) or None  # type: ignore
    allometry = str(get(CONSTANTS.ALLOMETRY_KEY, data, "chave dry"))

    if inputs is None:
        return []

    interventions = list(
        map(
            lambda intervention_input: handle_intervention(
                intervention_input=intervention_input,
                allometry=allometry,
                use_api=use_api,
                # Static placeholder values. Ideally these should come from the UI
                no_of_trees=3,
                create_forward_soil_model=ForwardSoilModel.create,
                create_inverse_soil_model=InverseSoilModel.create,
            ),
            inputs,
        )
    )

    return interventions
