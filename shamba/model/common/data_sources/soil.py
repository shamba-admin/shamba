import os
from osgeo import gdal
from typing import Tuple, List, Optional
import logging as log
from toolz import compose

from model.common import csv_handler
from rasters import soil as soil_raster

CSV_FILENAME = os.path.join(os.path.dirname(os.path.abspath(soil_raster.__file__)), "HWSD_data.csv")
BIL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(soil_raster.__file__)), "hwsd.bil")

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
    return data[0, 0] # MU_GLOBAL for input to HWSD_data.csv

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
        types=(int, int, float, int, "|S25", float, float, float, "|S15", float, float, float, float)
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
    return compose(get_data_from_identifier, get_identifier)(localtion)