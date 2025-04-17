import os
import numpy as np
import math
import calendar
from datetime import datetime, timedelta
from itertools import groupby
import requests
import socket
from typing import List, Any, Dict, Optional, Tuple

from model.common import csv_handler
from rasters import climate as climate_raster
from model.common.data_sources.helpers import return_none_on_exception

MONTHS_COUNT = 12

TEMPERATURE_BASENAME = "tmp_"
RAINFALL_BASENAME = "pre_"
PET_BASENAME = "pet_"
BASENAMES = [TEMPERATURE_BASENAME, RAINFALL_BASENAME, PET_BASENAME]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


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


def generate_dates(year: int) -> np.ndarray:
    start_date = datetime(year, 1, 1)
    return np.array(
        [start_date + timedelta(days=i) for i in range(365 + calendar.isleap(year))]
    )


def pair_dates_with_values(dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Ensure dates and values have the same length
    min_length = min(len(dates), len(values))
    return np.column_stack((dates[:min_length], values[:min_length]))


def group_by_month(date_value_pairs: np.ndarray) -> list:
    months = np.vectorize(lambda d: d.month)(date_value_pairs[:, 0])
    return [(month, date_value_pairs[months == month]) for month in range(1, 13)]


def calculate_monthly_average(
    month_group: Tuple[int, np.ndarray],
) -> Tuple[int, np.floating[Any]]:
    month, values = month_group
    return month, np.mean(values[:, 1])


def segment_and_average_by_month(daily_values: np.ndarray, year: int) -> np.ndarray:
    dates = generate_dates(year)
    date_value_pairs = pair_dates_with_values(dates, daily_values)
    grouped_by_month = group_by_month(date_value_pairs)
    monthly_averages = np.array(
        [calculate_monthly_average(group) for group in grouped_by_month]
    )
    return monthly_averages


def get_climate_data(longitude: float, latitude: float, use_api: bool) -> np.ndarray:
    """
    Get climate data for a given location.

    If use_api is True, use the API to get the data. Otherwise, use the local data.

    Each piece of climate data is a 12-month average. For the API response, the data is
    grouped by month, and then averaged over the months. Only the current year's data is
    used and extrapolated to previous years. This is not ideal, but it is the best we can given exact years are not available.

    Args:
        longitude (float): The longitude of the location.
        latitude (float): The latitude of the location.
        use_api (bool): Whether to use the API to get the data.

    Returns:
        np.ndarray: The climate data for the given location.
    """
    current_year = datetime.now().year
    previous_year = current_year - 1
    start_date = datetime(previous_year, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime(previous_year, 12, 31).strftime("%Y-%m-%d")

    api_response = (
        get_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            daily_params=[
                "temperature_2m_mean",
                "rain_sum",
                "et0_fao_evapotranspiration",
            ],
            start_date=start_date,
            end_date=end_date,
        )
        if use_api
        else None
    )

    if api_response is None:
        x = math.ceil(180 - 2 * latitude)
        y = math.ceil(360 + 2 * longitude)  # Fixed the multiplication
        folder = os.path.dirname(os.path.abspath(climate_raster.__file__))
        return populate_climate_data(folder, x, y)

    daily_data = api_response["daily"]
    temperature = segment_and_average_by_month(
        np.array(daily_data["temperature_2m_mean"]), previous_year
    )
    rain = segment_and_average_by_month(np.array(daily_data["rain_sum"]), previous_year)
    evapotranspiration = segment_and_average_by_month(
        np.array(daily_data["et0_fao_evapotranspiration"]), previous_year
    )

    result = np.vstack([temperature[:, 1], rain[:, 1], evapotranspiration[:, 1]])
    return result


@return_none_on_exception(requests.RequestException, socket.gaierror)
def get_weather_forecast(
    latitude: float,
    longitude: float,
    daily_params: List[str],
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily_params),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
        "format": "json",
        "timeformat": "unixtime",
    }

    print("IIII", params)

    response = requests.get(API_URL, params=params)
    return response.json()
