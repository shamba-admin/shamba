from datetime import datetime

LOCATION_LATITIUDE_KEY = "lat"
LOCATION_LONGITUDE_KEY = "lon"
NO_OF_YEARS_KEY = "yrs_proj"
NO_OF_TREES_KEY = "no_of_trees"
ALLOMETRY_KEY = "allometry"
# ----------
SPECIES_BASE_KEY = "species_base"
SPECIES_1_KEY = "species1"
# ----------
THINNING_BASE_YEAR_1_KEY = "thin_base_yr1"
THINNING_BASE_YEAR_2_KEY = "thin_base_yr2"
# ----------
THINNING_BASE_PC1_KEY = "thin_base_pc1"
THINNING_BASE_PC2_KEY = "thin_base_pc2"
# ----------
THINNING_PROJECT_YEAR_1_KEY = "thin_proj_yr1"
THINNING_PROJECT_YEAR_2_KEY = "thin_proj_yr2"
THINNING_PROJECT_YEAR_3_KEY = "thin_proj_yr3"
THINNING_PROJECT_YEAR_4_KEY = "thin_proj_yr4"
# ----------
THINNING_PROJECT_PC1_KEY = "thin_proj_pc1"
THINNING_PROJECT_PC2_KEY = "thin_proj_pc2"
THINNING_PROJECT_PC3_KEY = "thin_proj_pc3"
THINNING_PROJECT_PC4_KEY = "thin_proj_pc4"
# ----------
THINNING_BASE_BR_KEY = "thin_base_br"
THINNING_BASE_ST_KEY = "thin_base_st"
# ----------
THINNING_PROJECT_BR_KEY = "thin_proj_br"
THINNING_PROJECT_ST_KEY = "thin_proj_st"
# ----------
BASE_MORTALITY_KEY = "base_mort"
PROJECT_MORTALITY_KEY = "proj_mort"
# ----------
MORTALITY_BASE_BR_KEY = "mort_base_br"
MORTALITY_BASE_ST_KEY = "mort_base_st"
# ----------
MORTALITY_PROJECT_BR_KEY = "mort_proj_br"
MORTALITY_PROJECT_ST_KEY = "mort_proj_st"
# ----------
BASE_PLANT_DENSITY_KEY = "base_plant_dens"
# ----------
FIRE_INTERNAL_BASE_KEY = "fire_int_base"
FIRE_PRES_BASE_KEY = "fire_pres_base"
# ----------
FIRE_INTERNAL_PROJECT_KEY = "fire_int_proj"
FIRE_PRES_PROJECT_KEY = "fire_pres_proj"
# ----------
BASE_LITTER_INTERVAL_KEY = "base_lit_int"
BASE_LITTER_QUANTITY_KEY = "base_lit_qty"
# ----------
PROJECT_LITTER_INTERVAL_KEY = "proj_lit_int"
PROJECT_LITTER_QUANTITY_KEY = "proj_lit_qty"
# ----------
BASE_SYNTHETIC_FERTILISER_INTERVAL_KEY = "base_sf_int"
BASE_SYNTHETIC_FERTILISER_QUANTITY_KEY = "base_sf_qty"
BASE_SYNTHETIC_FERTILISER_N_KEY = "base_sf_n"
# ----------
PROJECT_SYNTHETIC_FERTILISER_INTERVAL_KEY = "proj_sf_int"
PROJECT_SYNTHETIC_FERTILISER_QUANTITY_KEY = "proj_sf_qty"
PROJECT_SYNTHETIC_FERTILISER_N_KEY = "proj_sf_n"
# ----------
BASE_COVER_MONTH_START_KEY = "base_cvr_mth_st"
BASE_COVER_MONTH_END_KEY = "base_cvr_mth_en"
BASE_COVER_PRES_KEY = "base_cvr_pres"
# ----------
PROJECT_COVER_MONTH_START_KEY = "proj_cvr_mth_st"
PROJECT_COVER_MONTH_END_KEY = "proj_cvr_mth_en"
PROJECT_COVER_PRES_KEY = "proj_cvr_pres"
# ----------
DEFAULT_NO_OF_TREES = 3
DEFAULT_ALLOMORPHY = "chave dry"
# ----------
DEFAULT_TREE_STANDARD_DENSITY = 100
DEFAULT_YEAR = datetime.now().year
DEFAULT_NO_OF_YEARS = 10

ROOT_IN_TOP_30 = 0.7 # TODO: check that this should always be the same for both crop and tree: it has been to date.

# Emissions stuff
# From table 2.5 IPCC 2006 GHG Inventory
ef = {"crop_N2O": 0.07, "crop_CH4": 2.7, "tree_N2O": 0.2, "tree_CH4": 6.8}
N_ef = 0.01  # emission factor [kbN20-N/kg N] # TODO: clarify where this is from and label better

# Global warming potential from IPCC Assessment Reports:

GWP_SAR = {"N2O": 310, "CH4": 21}
GWP_AR4 = {"N2O": 298, "CH4": 25}
GWP_AR5 = {"N2O": 265, "CH4": 28}

gwp = GWP_SAR # TODO: call this default_GWP, possibly change to AR5

# combustion factor from IPCC AFOLU table
cf = {"crop": 0.8, "tree": 0.74}

C_to_CO2_conversion_factor = 44.0 / 12
N_to_N2O_conversion_factor = 44.0 / 28  # for N2O-N to N2O

# Some parameters. See methodology
volatile_frac_synth = 0.1
volatile_frac_org = 0.2

# =============================================================================
#  Model configuration
# =============================================================================
DEFAULT_USE_API = True
ROOT_IN_TOP_30 = 0.7
