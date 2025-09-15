from typing import List, Dict, Any, Optional
from toolz import get
from dask.distributed import Client, LocalCluster
import dask.bag as db

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import handle_intervention
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
from model.soil_models.soil_model_types import SoilModelType

ForwardSoilModel = ForwardSoilModule.get_soil_model(SoilModelType.ROTH_C)
InverseSoilModel = InverseSoilModule.get_soil_model(SoilModelType.ROTH_C)

NO_OF_WORKERS = 4
THREADS_PER_WORKER = 2

def run(project_name, data, use_api: bool):
    """
    Run interventions in parallel using Dask.
    """
    inputs: Optional[List[Dict[str, Any]]] = get("inputs", data) or None
    allometry = str(get(CONSTANTS.ALLOMETRY_KEY, data, "chave dry"))

    if inputs is None:
        return []

    # Use context managers to manage Dask resources
    with LocalCluster(n_workers=NO_OF_WORKERS, threads_per_worker=THREADS_PER_WORKER) as cluster:
        with Client(cluster) as client:
            # Convert inputs to a Dask bag
            input_bag = db.from_sequence(inputs)

            results = input_bag.map(
                lambda intervention_input: handle_intervention(
                    intervention_input=intervention_input,
                    allometry=allometry,
                    use_api=use_api,
                    no_of_trees=3,
                    create_forward_soil_model=ForwardSoilModel.create,
                    create_inverse_soil_model=InverseSoilModel.create,
                )
            ).compute()  # Trigger computation and gather results

            return results
