from typing import Literal

import model.soil_models.forward_soil_model as roth_c
import model.soil_models.example_soil_model.forward_example as example_soil_model
from .soil_model_type import SoilModelType


def get_soil_model(soil_model_type: SoilModelType):
    match soil_type:
        case SoilModelType.RothC:
            return roth_c
        case SoilModelType.EXAMPLE:
            return example_soil_model
        case _:
            raise ValueError(f"Unknown soil model type: {soil_model_type}")


def print_to_stdout(forward_soil_model, no_of_years: int, label: str) -> None:
    """Print data from forward RothC run to stdout using tabulate with a functional approach."""
    table_title = f"FORWARD CALCULATIONS for {label}"

    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    soil_iom = forward_soil_model.soil.iom

    if len(tot_soc) == no_of_years + 1:
        years = np.arange(len(tot_soc), dtype=float)
    else:
        years = (
            np.arange(-len(tot_soc) + 2, 2, dtype=float) - forward_soil_model.Cy0Year
        )
        years[-1] = 0

    table_data = generate_table_data(
        tot_soc, soil_iom, forward_soil_model.inputs, years
    )

    headers = ["Year", "Carbon", "Crop In", "Tree In"]

    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="fancy_grid"))


def save(forward_soil_model, no_of_years, file="soil_model_forward.csv"):
    """Save data from forward RothC run to a csv.
    Default path is OUTPUT_DIR.

    """
    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    inputs = np.append(forward_soil_model.inputs, [[0, 0]], axis=0)
    data = np.column_stack(
        (
            tot_soc + forward_soil_model.soil.iom,
            forward_soil_model.SOC,
            np.array(len(tot_soc) * [forward_soil_model.soil.iom]),
            inputs[:, 0],
            inputs[:, 1],
        )
    )
    cols = ["soc", "dpm", "rpm", "bio", "hum", "iom", "crop_in", "tree_in"]
    if len(tot_soc) != no_of_years + 1:  # solve to value
        cols.insert(0, "year")
        x = np.array(list(range(-len(tot_soc) + 2, 2)))
        x = x - forward_soil_model.Cy0Year
        x[-1] = 0
        data = np.column_stack((x, data))
        csv_handler.print_csv(file, data, col_names=cols)
    else:
        csv_handler.print_csv(file, data, col_names=cols, print_years=True)


def plot(forward_soil_model, legend_string, no_of_years, save_name=None):
    """Plot total carbon vs year for forwardRothC run.

    Args:
        legend_string: string to put in legend

    """
    fig = plt.figure()
    fig.suptitle("Soil Carbon")  # Replace set_window_title with suptitle
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("SOC (t C ha^-1)")
    ax.set_title("Total soil carbon vs time")

    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    if len(tot_soc) == no_of_years + 1:
        # baseline or project
        x = list(range(len(tot_soc)))
    else:
        # initialisation run is before year 0
        x = np.array(list(range(-len(tot_soc) + 2, 2)))
        x = x - forward_soil_model.Cy0Year
        x[-1] = 0

    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    ax.plot(x, tot_soc, label=legend_string)
    ax.legend(loc="best")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))


def create_row(
    year: float, carbon: float, inputs: Optional[List[float]] = None
) -> List[float]:
    return [year, carbon] + (inputs or [])


def generate_table_data(
    tot_soc: np.ndarray,
    soil_iom: float,
    inputs: List[Tuple[float, float]],
    years: np.ndarray,
) -> List[List[float]]:
    return [
        create_row(year, soc + soil_iom, list(inputs[i]) if i < len(inputs) else [])
        for i, (year, soc) in enumerate(zip(years, tot_soc))
    ]
