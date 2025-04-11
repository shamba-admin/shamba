from toolz import get

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import handle_intervention


def run(project_name, data):
    inputs = get("inputs", data) or None
    allometry = str(get(CONSTANTS.ALLOMETRY_KEY, data, "chave dry"))

    if inputs is None:
        return []

    interventions = list(
        map(
            lambda intervention_input: handle_intervention(
                intervention_input=intervention_input,
                allometry=allometry,
            ),
            inputs,
        )
    )

    return interventions
