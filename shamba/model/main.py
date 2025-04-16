from typing import List, Dict, Any, Optional
from toolz import get

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import handle_intervention


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
            ),
            inputs,
        )
    )

    return interventions
