import os
from osgeo import gdal
from typing import Tuple, List, Optional, NamedTuple, Any, Dict
import logging as log
from toolz import compose
import requests
import socket
from enum import Enum
from copy import deepcopy
from statistics import mean

from model.common import csv_handler
from rasters import soil as soil_raster
from model.common.data_sources.helpers import return_none_on_exception

CSV_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(soil_raster.__file__)), "HWSD_data.csv"
)
BIL_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(soil_raster.__file__)), "hwsd.bil"
)

API_URL = "https://rest.isric.org/soilgrids/v2.0/"


def open_raster(filename: str) -> gdal.Dataset:
    """Open a raster file using GDAL."""
    gdal.AllRegister()
    driver = gdal.GetDriverByName("HFA")
    driver.Register()
    gdal.UseExceptions()

    try:
        return gdal.Open(filename)
    except RuntimeError:
        raise csv_handler.FileOpenError(filename)


def get_raster_value(ds: gdal.Dataset, x: float, y: float) -> int:
    """Get the raster value at a given location."""
    # Georeference information
    transform = ds.GetGeoTransform()
    x_origin, y_origin = transform[0], transform[3]
    width, height = transform[1], transform[5]

    x_int = int((x - x_origin) / width)
    y_int = int((y - y_origin) / height)

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(x_int, y_int, 1, 1)
    return data[0, 0]  # MU_GLOBAL for input to HWSD_data.csv


def get_identifier(location: Tuple[float, float]) -> int:
    """Find MU_GLOBAL for given location from the HWSD .bil raster."""
    y, x = location

    ds = open_raster(BIL_FILENAME)
    return get_raster_value(ds, x, y)


def read_soil_table(filename: str) -> List[Tuple]:
    """Read the soil data from CSV file."""
    return csv_handler.read_mixed_csv(
        filename,
        cols=tuple(range(13)),
        types=(
            int,
            int,
            float,
            int,
            "|S25",
            float,
            float,
            float,
            "|S15",
            float,
            float,
            float,
            float,
        ),
    )


def filter_rows_by_mu(soil_table: List[Tuple], mu: int) -> List[Tuple]:
    """Filter rows from soil table by MU_GLOBAL value."""
    return [row for row in soil_table if row[1] == mu]


def calculate_weighted_sum(mu_rows: List[Tuple]) -> Tuple[float, float]:
    """Calculate weighted sum of SOC and clay from soil data rows."""
    cy0 = sum(row[12] * row[2] for row in mu_rows) / 100
    clay = sum(row[7] * row[2] for row in mu_rows) / 100
    return cy0, clay


def get_data_from_identifier(mu: int) -> Optional[Tuple[float, float]]:
    """Get soil data from csv given MU_GLOBAL from the raster."""
    soil_table = read_soil_table(CSV_FILENAME)
    mu_rows = filter_rows_by_mu(soil_table, mu)

    if not mu_rows:
        log.warning("COULD NOT FIND %d IN HWSD_DATA.csv", mu)
        return None

    return calculate_weighted_sum(mu_rows)


def get_soil_data(localtion) -> Optional[Tuple[float, float]]:
    """Get soil data from soilgrids api.
    If data is not available, get data from local csv file.
    """
    api_response = get_properties_from_soilgrids_api(
        location=Point(localtion[1], localtion[0])
    )

    if api_response is None:
        return compose(get_data_from_identifier, get_identifier)(localtion)

    return compose(
        get_soc_and_clay,
        process_data,
        convert_units_in_api_response,
    )(api_response)


class SoilProperty(Enum):
    """Enum for Soil Grids soil properties.

    This class defines properties available in Soil Grids.

    """

    BULK_DENSITY_OF_FINE_FRACTION = "bdod"
    CATION_EXCHANGE_CAPACITY = "cec"
    VOLUMETRIC_FRACTION_OF_COARSE_FRACTION = "cfvo"
    PROPORTION_OF_CLAY_IN_FINE_FRACTION = "clay"
    TOTAL_NITROGEN = "nitrogen"
    PH = "phh2o"
    ORGANIC_CARBON_DENSITY = "ocd"
    ORGANIC_CARBON_STOCKS = "ocs"
    PROPORTION_OF_SAND_IN_FINE_FRACTION = "sand"
    PROPORTION_OF_SILT_IN_FINE_FRACTION = "silt"
    SOIL_ORGANIC_CARBON_CONTENT_IN_FINE_FRACTION = "soc"
    VOL_WATER_CONTENT_MINUS_10KPA = "wv0010"
    VOL_WATER_CONTENT_MINUS_33KPA = "wv0033"
    VOL_WATER_CONTENT_MINUS_1500KPA = "wv1500"


DEFAULT_SOIL_PROPERTIES = [
    SoilProperty.PROPORTION_OF_CLAY_IN_FINE_FRACTION,
    SoilProperty.SOIL_ORGANIC_CARBON_CONTENT_IN_FINE_FRACTION,
    SoilProperty.PROPORTION_OF_SILT_IN_FINE_FRACTION,
    SoilProperty.PROPORTION_OF_SAND_IN_FINE_FRACTION,
]


class Depth(Enum):
    """Enum for Soil Grids depths.

    This class defines depths available in Soil Grids.

    """

    ZERO_TO_FIVE_CM = "0-5cm"
    ZERO_TO_THIRTY_CM = "0-30cm"
    FIVE_TO_FIFTEEN_CM = "5-15cm"
    FIFTEEN_TO_THIRTY_CM = "15-30cm"
    THIRTY_TO_SIXTY_CM = "30-60cm"
    SIXTY_TO_HUNDRED_CM = "60-100cm"
    ONE_HUNDRED_TO_TWO_HUNDRED_CM = "100-200cm"


DEFAULT_DEPTHS = [
    Depth.ZERO_TO_FIVE_CM,
    Depth.FIVE_TO_FIFTEEN_CM,
    Depth.FIFTEEN_TO_THIRTY_CM,
]


class Value(Enum):
    """Enum for Soil Grids values.

    This class defines values available in Soil Grids.

    """

    Q0_05 = "Q0.05"
    Q0_5 = "Q0.5"
    Q0_95 = "Q0.95"
    MEAN = "mean"
    UNCERTAINTY = "uncertainty"


DEFAULT_VALUES = [Value.MEAN, Value.Q0_05, Value.Q0_95]

# Conversion to conventional units from here:
# https://www.isric.org/explore/soilgrids/faq-soilgrids (properties table)
UNIT_CONVERSIONS = {
    SoilProperty.BULK_DENSITY_OF_FINE_FRACTION: 100,
    SoilProperty.CATION_EXCHANGE_CAPACITY: 10,
    SoilProperty.VOLUMETRIC_FRACTION_OF_COARSE_FRACTION: 10,
    SoilProperty.PROPORTION_OF_CLAY_IN_FINE_FRACTION: 10,
    SoilProperty.TOTAL_NITROGEN: 100,
    SoilProperty.PH: 10,
    SoilProperty.ORGANIC_CARBON_DENSITY: 10,
    SoilProperty.ORGANIC_CARBON_STOCKS: 10,
    SoilProperty.PROPORTION_OF_SAND_IN_FINE_FRACTION: 10,
    SoilProperty.PROPORTION_OF_SILT_IN_FINE_FRACTION: 10,
    SoilProperty.SOIL_ORGANIC_CARBON_CONTENT_IN_FINE_FRACTION: 10,
    SoilProperty.VOL_WATER_CONTENT_MINUS_10KPA: 1,
    SoilProperty.VOL_WATER_CONTENT_MINUS_33KPA: 1,
    SoilProperty.VOL_WATER_CONTENT_MINUS_1500KPA: 1,
}


# Type aliases for results
DepthValueDict = dict[str, dict[str, float]]
SoilPropertyDepthValueDict = dict[SoilProperty, DepthValueDict]


class Point(NamedTuple):
    longitude: float
    latitude: float


@return_none_on_exception(requests.RequestException, socket.gaierror)
def get_properties_from_soilgrids_api(
    location: Point,
    soil_properties: list[SoilProperty] = DEFAULT_SOIL_PROPERTIES,
    depths: list[Depth] = DEFAULT_DEPTHS,
    values: list[Value] = DEFAULT_VALUES,
) -> Optional[dict]:
    """Retrieve soil property data for a given location.

    This function queries the Soil Grids API to fetch soil properties based on specified
    coordinates, soil property types, depths, and value types.

    Args:
        location (Point): A Point object containing longitude and latitude coordinates.
        soil_properties (list[SoilProperty], optional): A list of soil properties to retrieve.
            Defaults to DEFAULT_SOIL_PROPERTIES.
        depths (list[Depth], optional): A list of depth levels to query.
            Defaults to DEFAULT_DEPTHS.
        values (list[Value], optional): A list of value types (e.g., mean, median) to retrieve.
            Defaults to DEFAULT_VALUES.

    Returns
    -------
        Optional[dict]: A dictionary containing the queried soil properties from the API.

    """
    query_url = API_URL + "properties/query"
    str_soil_properties = [soil_property.value for soil_property in soil_properties]
    str_depths = [depth.value for depth in depths]
    str_values = [value.value for value in values]
    params = {
        "lon": location.longitude,
        "lat": location.latitude,
        "property": str_soil_properties,
        "depth": str_depths,
        "value": str_values,
    }
    headers = {"accept": "application/json"}
    response = requests.get(query_url, params=params, headers=headers)
    return response.json()


def get_soc_and_clay(api_response: List[Tuple[str, float]]) -> Tuple[float, float]:
    return next((value for name, value in api_response if name == "soc"), 0.0), next(
        (value for name, value in api_response if name == "clay"), 0.0
    )


def process_data(api_response: Dict[str, Any]) -> List[Tuple[str, float]]:
    data = api_response["properties"]["layers"]
    return [
        (layer["name"], mean(depth["values"]["mean"] for depth in layer["depths"]))
        for layer in data
    ]


def convert_value(value: float, conversion_factor: float) -> float:
    return value / conversion_factor if value != 0 else value


def convert_units_in_api_response(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert units in the Soil Grids API response using predefined conversion factors.

    This function creates a new dictionary by applying unit conversions to soil property
    values based on predefined conversion factors.

    https://www.isric.org/explore/soilgrids/faq-soilgrids (properties table)

    Args:
        d (dict): The Soil Grids API response containing soil property data with unit-dependent values.

    Returns:
        dict: A new dictionary with converted values.
    """
    result = deepcopy(api_response)
    layers = result.get("properties", {}).get("layers", [])

    for layer in layers:
        name_enum = SoilProperty(layer.get("name", ""))
        conversion_factor = UNIT_CONVERSIONS.get(name_enum, 1)

        for depth in layer.get("depths", []):
            depth["values"] = {
                k: convert_value(v, conversion_factor)
                for k, v in depth.get("values", {}).items()
            }

    return result
