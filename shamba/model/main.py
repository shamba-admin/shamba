from toolz import get

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import get_int, handle_intervention


def run(project_name, data):
    inputs = get("inputs", data, None)
    no_of_trees = get_int(
        CONSTANTS.NO_OF_TREES_KEY, data, CONSTANTS.DEFAULT_NO_OF_TREES
    )
    allometry = get(CONSTANTS.ALLOMETRY_KEY, data, "chave dry")

    interventions = list(
        map(
            lambda intervention_input: handle_intervention(
                intervention_input=intervention_input,
                no_of_trees=no_of_trees,
                allometry=allometry,
            ),
            inputs,
        )
    )

    return interventions
