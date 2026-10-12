"""Crop a global ECMWF weather model to an area of interest (AOI).

The ASF_ECMWF_TROP files isce3 uses for its troposphere correction are global
(~2 GB compressed), but isce3 only reads the model nodes inside the radar grid
plus a small margin. Slicing the model to the AOI before staging it gives the
same delays from a file a few MB in size.
"""

import numpy as np
import xarray as xr


def crop_tropo(ds: xr.Dataset, subset: list[float], buffer: float = 1.0) -> xr.Dataset:
    """Slice a global ECMWF weather model to an AOI.

    isce3 (via RAiDER) only reads model nodes inside the radar grid plus a 0.1 degree
    margin, so a crop covering the AOI plus the buffer gives identical delays.

    Args:
        ds: ECMWF dataset with descending `latitude` and 0-360 `longitude` axes.
        subset: AOI [lon_min, lat_min, lon_max, lat_max] in degrees.
        buffer: Padding added on every side of the AOI, in degrees.

    Returns:
        The cropped dataset.
    """
    lon_min, lat_min, lon_max, lat_max = subset
    lat = ds.latitude.values
    lon = ds.longitude.values
    # Latitude is stored north to south, so the AOI rows are one contiguous run of indices.
    rows = np.where((lat >= lat_min - buffer) & (lat <= lat_max + buffer))[0]
    lat_slice = slice(rows[0], rows[-1] + 1)

    # The model's longitudes run 0-360, so map the AOI edges onto that axis.
    west, east = (lon_min - buffer) % 360, (lon_max + buffer) % 360
    if west <= east:
        cols = np.where((lon >= west) & (lon <= east))[0]
        return ds.isel(latitude=lat_slice, longitude=slice(cols[0], cols[-1] + 1))
    # The AOI straddles 0 degrees longitude: join the piece west of 360 to the piece east of 0.
    west_cols = np.where(lon >= west)[0]
    east_cols = np.where(lon <= east)[0]
    west_part = ds.isel(latitude=lat_slice, longitude=slice(west_cols[0], None))
    east_part = ds.isel(latitude=lat_slice, longitude=slice(None, east_cols[-1] + 1))
    return xr.concat([west_part, east_part], dim='longitude')
