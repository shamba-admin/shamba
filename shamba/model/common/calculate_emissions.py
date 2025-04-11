from typing import Dict, Union, List, NamedTuple, Tuple, Any, Callable
from toolz import get, compose  # type: ignore
import numpy as np

import model.command_line.climate as Climate
import model.command_line.crop_model as CropModel
import model.command_line.litter as LitterModel
import model.command_line.soil_models.roth_c.forward_roth_c as ForwardRothC
import model.command_line.soil_models.roth_c.inverse_roth_c as InverseRothC
import model.command_line.soil_params as SoilParams
import model.command_line.tree_growth as TreeGrowth
import model.command_line.tree_model as TreeModel
import model.command_line.tree_params as TreeParams
import model.command_line.crop_params as CropParams
import model.command_line.emit as Emit
import model.common.constants as CONSTANTS

get_float: Callable[[str, Dict[str, Any]], float] = compose(float, get)  # type: ignore
get_int: Callable[[str, Dict[str, Any]], int] = compose(int, get)  # type: ignore


def get_location(year_input: Dict[str, Any]) -> Tuple[float, float]:
    print("LLLLLL", get_float(CONSTANTS.LOCATION_LATITIUDE_KEY, year_input))
    return (
        get_float(CONSTANTS.LOCATION_LATITIUDE_KEY, year_input),
        get_float(CONSTANTS.LOCATION_LONGITUDE_KEY, year_input),
    )


def populate_thinning_array(
    intervention_input: Dict[str, Union[float, int]],
    key_pairs: List[Tuple[str, str]],
    no_of_years: int,
):
    thinning_array = np.zeros(no_of_years + 1)
    for year_key, percent_key in key_pairs:
        year = get_int(year_key, intervention_input)
        percent = get_float(percent_key, intervention_input)
        thinning_array[year] = percent
    return thinning_array


# ----------
# TREE MODEL
# ----------
class GetTreeModelReturnData(NamedTuple):
    tree_base: TreeModel.TreeModel
    tree_projects: List[TreeModel.TreeModel]


def get_tree_model_data(
    intervention_input: Dict[str, Union[float, int]],
    no_of_years: int,
    no_of_trees: int,
    allometry: str,
) -> GetTreeModelReturnData:
    # Linking tree cohort parameteres
    tree_par_base = TreeParams.from_species_index(
        get_int(CONSTANTS.SPECIES_BASE_KEY, intervention_input)
    )
    tree_params_1 = TreeParams.from_species_index(
        get_int(CONSTANTS.SPECIES_1_KEY, intervention_input)
    )

    tree_params = TreeParams.create_tree_params_from_species_index(
        intervention_input, no_of_trees
    )

    # Linking tree growth
    growth_base = TreeGrowth.get_growth(
        intervention_input,
        CONSTANTS.SPECIES_BASE_KEY,
        tree_par_base,
        allometric_key=allometry,
    )

    tree_growths = TreeGrowth.create_tree_growths(
        intervention_input, tree_params, allometry, no_of_trees
    )

    # Specify thinning regime and fraction left in field (lif)
    # baseline thinning regime
    # (add line of thinning[yr] = % thinned for each event)
    thinning_base = populate_thinning_array(
        intervention_input,
        [
            (CONSTANTS.THINNING_BASE_YEAR_1_KEY, CONSTANTS.THINNING_BASE_PC1_KEY),
            (CONSTANTS.THINNING_BASE_YEAR_2_KEY, CONSTANTS.THINNING_BASE_PC2_KEY),
        ],
        no_of_years,
    )

    thinning_project = populate_thinning_array(
        intervention_input,
        [
            (CONSTANTS.THINNING_PROJECT_YEAR_1_KEY, CONSTANTS.THINNING_PROJECT_PC1_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_2_KEY, CONSTANTS.THINNING_PROJECT_PC2_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_3_KEY, CONSTANTS.THINNING_PROJECT_PC3_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_4_KEY, CONSTANTS.THINNING_PROJECT_PC4_KEY),
        ],
        no_of_years,
    )

    # Baseline fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_base = np.array(
        [
            1,
            get_float(CONSTANTS.THINNING_BASE_BR_KEY, intervention_input),
            get_float(CONSTANTS.THINNING_BASE_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Project fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_project = np.array(
        [
            1,
            get_float(CONSTANTS.THINNING_PROJECT_BR_KEY, intervention_input),
            get_float(CONSTANTS.THINNING_PROJECT_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Specify mortality regime and fraction left in field (lif)
    # Baseline yearly mortality
    mortality_base = np.array(
        (no_of_years + 1)
        * [get_float(CONSTANTS.BASE_MORTALITY_KEY, intervention_input)]
    )

    # Project yearly mortality
    mortality_project = np.array(
        (no_of_years + 1)
        * [get_float(CONSTANTS.PROJECT_MORTALITY_KEY, intervention_input)]
    )

    # Baseline fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_base = np.array(
        [
            1,
            get_float(CONSTANTS.MORTALITY_BASE_BR_KEY, intervention_input),
            get_float(CONSTANTS.MORTALITY_BASE_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Project fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_project = np.array(
        [
            1,
            get_float(CONSTANTS.MORTALITY_PROJECT_BR_KEY, intervention_input),
            get_float(CONSTANTS.MORTALITY_PROJECT_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # RUN TREE MODEL
    # Trees planted in baseline (standDens must be at least 1)
    tree_base = TreeModel.from_defaults(
        tree_params=tree_params_1,
        tree_growth=growth_base,
        yearPlanted=0,
        standard_density=get_int(CONSTANTS.BASE_PLANT_DENSITY_KEY, intervention_input)
        or CONSTANTS.DEFAULT_TREE_STANDARD_DENSITY,
        thin=thinning_base,
        thinFrac=thinning_fraction_left_base,
        mort=mortality_base,
        mortFrac=mortality_fraction_left_base,
        no_of_years=no_of_years,
    )

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=intervention_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinning_project=thinning_project,
        thinning_fraction_left_project=thinning_fraction_left_project,
        mortality_project=mortality_project,
        mortality_fraction_left_project=mortality_fraction_left_project,
        no_of_years=no_of_years,
        tree_count=no_of_trees,
    )

    return GetTreeModelReturnData(tree_base=tree_base, tree_projects=tree_projects)


# ----------
# FIRE MODEL
# ----------
class GetFireModelReturnData(NamedTuple):
    fire_base: np.ndarray
    fire_project: np.ndarray


def get_fire_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetFireModelReturnData:
    # Return interval of fire, [::2] = 1 is return interval of two years
    base_fire_interval = get_int(CONSTANTS.FIRE_INTERNAL_BASE_KEY, intervention_input)
    if base_fire_interval == 0:
        fire_base = np.zeros(no_of_years)
    else:
        fire_base = np.zeros(no_of_years)
        fire_base[::base_fire_interval] = get_int(
            CONSTANTS.FIRE_PRES_BASE_KEY, intervention_input
        )

    project_fire_interval = get_int(
        CONSTANTS.FIRE_INTERNAL_PROJECT_KEY, intervention_input
    )
    if project_fire_interval == 0:
        fire_project = np.zeros(no_of_years)
    else:
        fire_project = np.zeros(no_of_years)
        fire_project[::project_fire_interval] = get_int(
            CONSTANTS.FIRE_PRES_PROJECT_KEY, intervention_input
        )
    return GetFireModelReturnData(fire_base=fire_base, fire_project=fire_project)


class GetLitterModelReturnData(NamedTuple):
    litter_external_base: LitterModel.LitterModelData
    litter_external_project: LitterModel.LitterModelData
    synthetic_fertiliser_base: LitterModel.LitterModelData
    synthetic_fertiliser_project: LitterModel.LitterModelData


def get_litter_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetLitterModelReturnData:
    # baseline external organic inputs
    litter_external_base = LitterModel.from_defaults(
        litterFreq=get_int(CONSTANTS.BASE_LITTER_INTERVAL_KEY, intervention_input),
        litterQty=get_float(CONSTANTS.BASE_LITTER_QUANTITY_KEY, intervention_input),
        no_of_years=no_of_years,
    )

    # baseline synthetic fertiliser additions
    synthetic_fertiliser_base = LitterModel.synthetic_fert(
        freq=get_int(
            CONSTANTS.BASE_SYNTHETIC_FERTILISER_INTERVAL_KEY, intervention_input
        ),
        qty=get_float(
            CONSTANTS.BASE_SYNTHETIC_FERTILISER_QUANTITY_KEY, intervention_input
        ),
        nitrogen=get_float(
            CONSTANTS.BASE_SYNTHETIC_FERTILISER_N_KEY, intervention_input
        ),
        no_of_years=no_of_years,
    )

    # Project external organic inputs
    litter_external_project = LitterModel.from_defaults(
        litterFreq=get_int(CONSTANTS.PROJECT_LITTER_INTERVAL_KEY, intervention_input),
        litterQty=get_float(CONSTANTS.PROJECT_LITTER_QUANTITY_KEY, intervention_input),
        no_of_years=no_of_years,
    )

    # Project synthetic fertiliser additions
    synthetic_fertiliser_project = LitterModel.synthetic_fert(
        freq=get_int(
            CONSTANTS.PROJECT_SYNTHETIC_FERTILISER_INTERVAL_KEY, intervention_input
        ),
        qty=get_float(
            CONSTANTS.PROJECT_SYNTHETIC_FERTILISER_QUANTITY_KEY, intervention_input
        ),
        nitrogen=get_float(
            CONSTANTS.PROJECT_SYNTHETIC_FERTILISER_N_KEY, intervention_input
        ),
        no_of_years=no_of_years,
    )

    return GetLitterModelReturnData(
        litter_external_base=litter_external_base,
        litter_external_project=litter_external_project,
        synthetic_fertiliser_base=synthetic_fertiliser_base,
        synthetic_fertiliser_project=synthetic_fertiliser_project,
    )


class GetCropModelReturnData(NamedTuple):
    crop_base: List[CropModel.CropModelData]
    crop_par_base: List[CropParams.CropParamsData]
    crop_project: List[CropModel.CropModelData]
    crop_par_project: List[CropParams.CropParamsData]


def get_crop_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetCropModelReturnData:
    crop_base, crop_par_base = CropModel.get_crop_bases(
        input_data=intervention_input,
        no_of_years=no_of_years,
        start_index=1,
        end_index=3,
    )
    crop_project, crop_par_project = CropModel.get_crop_projects(
        input_data=intervention_input,
        no_of_years=no_of_years,
        start_index=1,
        end_index=3,
    )

    return GetCropModelReturnData(
        crop_base=crop_base,
        crop_par_base=crop_par_base,
        crop_project=crop_project,
        crop_par_project=crop_par_project,
    )


class GetSoilCarbonReturnData(NamedTuple):
    roth_base: ForwardRothC.ForwardRothCData
    roth_project: ForwardRothC.ForwardRothCData


def get_soil_carbon_data(
    intervention_input: Dict[str, Union[float, int]],
    no_of_years: int,
    climate: Climate.ClimateData,
    soil: SoilParams.SoilParamsData,
    inverse_roth: InverseRothC.InverseRothCData,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
) -> GetSoilCarbonReturnData:
    # soil cover for baseline
    cover_base = np.zeros(12)
    cover_base[
        get_int(CONSTANTS.BASE_COVER_MONTH_START_KEY, intervention_input) : get_int(
            CONSTANTS.BASE_COVER_MONTH_END_KEY, intervention_input
        )
    ] = get_int(CONSTANTS.BASE_COVER_PRES_KEY, intervention_input)

    # soil cover for project
    cover_proj = np.zeros(12)

    cover_proj[
        get_int(CONSTANTS.PROJECT_COVER_MONTH_START_KEY, intervention_input) : get_int(
            CONSTANTS.PROJECT_COVER_MONTH_END_KEY, intervention_input
        )
    ] = get_int(CONSTANTS.PROJECT_COVER_PRES_KEY, intervention_input)

    # Solve to y=0
    for_roth = ForwardRothC.create(
        soil,
        climate,
        cover_base,
        no_of_years=no_of_years,
        Ci=inverse_roth.eqC,
        crop=crop_base,
        fire=fire_base,
        solveToValue=True,
    )

    # Soil carbon for baseline and project
    roth_base = ForwardRothC.create(
        soil=soil,
        climate=climate,
        cover=cover_base,
        Ci=for_roth.SOC[-1],
        no_of_years=no_of_years,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fire=fire_base,
    )

    roth_project = ForwardRothC.create(
        soil,
        climate,
        cover_proj,
        Ci=for_roth.SOC[-1],
        no_of_years=no_of_years,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fire=fire_project,
    )

    return GetSoilCarbonReturnData(roth_base=roth_base, roth_project=roth_project)


class GetEmissionsReturnData(NamedTuple):
    emit_base_emissions: np.ndarray
    emit_project_emissions: np.ndarray


def get_emissions_data(
    no_of_years: int,
    roth_base: ForwardRothC.ForwardRothCData,
    roth_project: ForwardRothC.ForwardRothCData,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
    synthetic_fertiliser_base: LitterModel.LitterModelData,
    synthetic_fertiliser_project: LitterModel.LitterModelData,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
) -> GetEmissionsReturnData:
    # Emissions stuff
    emit_base_emissions = Emit.create(
        no_of_years=no_of_years,
        forRothC=roth_base,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fert=[synthetic_fertiliser_base],
        fire=fire_base,
    )
    emit_project_emissions = Emit.create(
        no_of_years=no_of_years,
        forRothC=roth_project,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fert=[synthetic_fertiliser_project],
        fire=fire_project,
    )

    return GetEmissionsReturnData(
        emit_base_emissions=emit_base_emissions,
        emit_project_emissions=emit_project_emissions,
    )


class GetEmissionsWithDifferenceReturnData(NamedTuple):
    base_emissions: np.ndarray
    project_emissions: np.ndarray
    difference: np.ndarray


def get_crop_emissions(
    no_of_years: int,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    fire_base: np.ndarray,
    fire_project: np.ndarray,
) -> GetEmissionsWithDifferenceReturnData:
    crop_base_emissions = Emit.create(
        no_of_years=no_of_years, crop=crop_base, fire=fire_base
    )
    crop_project_emissions = Emit.create(
        no_of_years=no_of_years, crop=crop_project, fire=fire_project
    )
    crop_difference = crop_project_emissions - crop_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=crop_base_emissions,
        project_emissions=crop_project_emissions,
        difference=crop_difference,
    )


def get_fertiliser_emissions(
    no_of_years: int,
    synthetic_fertiliser_base: LitterModel.LitterModelData,
    synthetic_fertiliser_project: LitterModel.LitterModelData,
) -> GetEmissionsWithDifferenceReturnData:
    fertiliser_base_emissions = Emit.create(
        no_of_years=no_of_years, fert=[synthetic_fertiliser_base]
    )
    fertiliser_project_emissions = Emit.create(
        no_of_years=no_of_years, fert=[synthetic_fertiliser_project]
    )
    fertiliser_difference = fertiliser_project_emissions - fertiliser_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=fertiliser_base_emissions,
        project_emissions=fertiliser_project_emissions,
        difference=fertiliser_difference,
    )


def get_litter_emissions(
    no_of_years: int,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
) -> GetEmissionsWithDifferenceReturnData:
    litter_base_emissions = Emit.create(
        no_of_years=no_of_years, litter=[litter_external_base], fire=fire_base
    )
    litter_project_emissions = Emit.create(
        no_of_years=no_of_years, litter=[litter_external_project], fire=fire_project
    )
    litter_difference = litter_project_emissions - litter_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=litter_base_emissions,
        project_emissions=litter_project_emissions,
        difference=litter_difference,
    )


def get_fire_emissions(
    no_of_years: int, fire_base: np.ndarray, fire_project: np.ndarray
) -> GetEmissionsWithDifferenceReturnData:
    fire_base_emissions = Emit.create(no_of_years=no_of_years, fire=fire_base)
    fire_project_emissions = Emit.create(no_of_years=no_of_years, fire=fire_project)
    fire_difference = fire_project_emissions - fire_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=fire_base_emissions,
        project_emissions=fire_project_emissions,
        difference=fire_difference,
    )


def get_tree_emissions(
    no_of_years: int,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
) -> GetEmissionsWithDifferenceReturnData:
    tree_base_emissions = Emit.create(
        no_of_years=no_of_years, tree=[tree_base], fire=fire_base
    )
    tree_project_emissions = Emit.create(
        no_of_years=no_of_years,
        tree=tree_projects,
        fire=fire_project,
    )
    tree_difference = tree_project_emissions - tree_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=tree_base_emissions,
        project_emissions=tree_project_emissions,
        difference=tree_difference,
    )


def handle_intervention(
    intervention_input: Dict[str, Union[float, int]],
    allometry: str = CONSTANTS.DEFAULT_ALLOMORPHY,
    no_of_trees: int = CONSTANTS.DEFAULT_NO_OF_TREES,
):
    no_of_years = (
        get_int(CONSTANTS.NO_OF_YEARS_KEY, intervention_input)
        or CONSTANTS.DEFAULT_NO_OF_YEARS
    )

    # ----------
    # LOCATION INFORMATION
    # ----------
    location = get_location(intervention_input)
    climate = Climate.from_location(location)

    # ----------
    # SOIL EQUILIBRIUM SOLVE
    # ----------
    soil = SoilParams.from_location(location)
    inverse_roth = InverseRothC.create(soil, climate)

    # ----------
    # MODEL DATA
    # ----------
    crop_model_data = get_crop_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
    )

    fire_model_data = get_fire_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
    )

    litter_model_data = get_litter_model_data(
        no_of_years=no_of_years, intervention_input=intervention_input
    )

    tree_model_data = get_tree_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
        no_of_trees=no_of_trees,
        allometry=allometry,
    )

    # ----------
    # EMISSIONS
    # ----------
    crop_emissions = get_crop_emissions(
        no_of_years=no_of_years,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
    )

    fertiliser_emissions = get_fertiliser_emissions(
        no_of_years=no_of_years,
        synthetic_fertiliser_base=litter_model_data.synthetic_fertiliser_base,
        synthetic_fertiliser_project=litter_model_data.synthetic_fertiliser_project,
    )

    litter_emissions = get_litter_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
    )

    fire_emissions = get_fire_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
    )

    tree_emissions = get_tree_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
    )

    # ----------
    # SOIL EMISSIONS
    # ----------
    soil_carbon_data = get_soil_carbon_data(
        intervention_input=intervention_input,
        no_of_years=no_of_years,
        climate=climate,
        soil=soil,
        inverse_roth=inverse_roth,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
    )

    emissions = get_emissions_data(
        no_of_years=no_of_years,
        roth_base=soil_carbon_data.roth_base,
        roth_project=soil_carbon_data.roth_project,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
        synthetic_fertiliser_base=litter_model_data.synthetic_fertiliser_base,
        synthetic_fertiliser_project=litter_model_data.synthetic_fertiliser_project,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
    )

    soil_base_emissions = emissions.emit_base_emissions - (
        crop_emissions.base_emissions
        + fertiliser_emissions.base_emissions
        + litter_emissions.base_emissions
        + fire_emissions.base_emissions
        + tree_emissions.base_emissions
    )

    soil_project_emissions = emissions.emit_project_emissions - (
        crop_emissions.project_emissions
        + fertiliser_emissions.project_emissions
        + litter_emissions.project_emissions
        + fire_emissions.project_emissions
        + tree_emissions.project_emissions
    )

    soil_difference = soil_project_emissions - soil_base_emissions

    result = {
        "soil_base_emissions": soil_base_emissions,
        "soil_project_emissions": soil_project_emissions,
        "soil_difference": soil_difference,
    }

    return result
