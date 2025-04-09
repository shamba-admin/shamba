import os
import numpy as np
import math
import requests
import socket
from typing import List, Any, Dict, Optional

from model.common import csv_handler
from rasters import climate as climate_raster
from model.common.data_sources.helpers import return_none_on_exception

MONTHS_COUNT = 12

TEMPERATURE_BASENAME = "tmp_"
RAINFALL_BASENAME = "pre_"
PET_BASENAME = "pet_"
BASENAMES = [TEMPERATURE_BASENAME, RAINFALL_BASENAME, PET_BASENAME]

API_URL = "https://api.open-meteo.com/v1/forecast"


def get_raster_filepath(folder: str, basename: str, month: int) -> str:
    return os.path.join(folder, f"{basename}{month}.txt")


def read_raster_value(filepath: str, x: int, y: int) -> float:
    try:
        return np.loadtxt(filepath, usecols=[y - 1], skiprows=x + 5)[0]
    except IOError:
        raise csv_handler.FileOpenError(filepath)


def populate_climate_data(folder: str, x: int, y: int) -> np.ndarray:
    climate_data = np.zeros((len(BASENAMES), MONTHS_COUNT))

    for k, basename in enumerate(BASENAMES):
        for month in range(1, MONTHS_COUNT + 1):
            filepath = get_raster_filepath(folder, basename, month)
            climate_data[k, month - 1] = read_raster_value(filepath, x, y)

    return climate_data


def get_climate_data(longitude: float, latitude: float) -> np.ndarray:
    # api_response = get_weather_forecast(latitude, longitude, ["temperature_2m_mean", "rain_sum", "evapotranspiration"])
    # print("XXXXX", api_response)
    api_response = None

    if api_response is None:
        # Indices for picking out climate data from rasters
        x = math.ceil(180 - 2 * latitude)
        # TODO: Is this a bug. Is mulitplying by `int` with no arguments always returns 0?
        y = math.ceil(360 + 2)
        folder = os.path.dirname(os.path.abspath(climate_raster.__file__))
        return populate_climate_data(folder, x, y)

    return api_response


@return_none_on_exception(requests.RequestException, socket.gaierror)
def get_weather_forecast(
    latitude: float, longitude: float, hourly_params: List[str]
) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(hourly_params),
    }

    response = requests.get(API_URL, params=params)
    return response.json()
