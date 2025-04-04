import os
import numpy as np
from typing import List

from model.common import csv_handler
from rasters import climate as climate_raster

MONTHS_COUNT = 12

TEMPERATURE_BASENAME = "tmp_"
RAINFALL_BASENAME = "pre_"
PET_BASENAME = "pet_"
BASENAMES = [TEMPERATURE_BASENAME, RAINFALL_BASENAME, PET_BASENAME]

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

def get_climate_data(x: int, y: int) -> np.ndarray:
    folder = os.path.dirname(os.path.abspath(climate_raster.__file__))
    
    return populate_climate_data(folder, x, y)