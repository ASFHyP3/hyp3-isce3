import numpy as np
import xarray as xr

from hyp3_isce3.crop_tropo import crop_tropo


def _global_model():
    # 1-degree global grid with the ECMWF file's axis conventions: latitude north to south, longitude 0-360.
    lat = np.arange(90.0, -91.0, -1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    t = np.arange(lat.size * lon.size, dtype=float).reshape(1, 1, lat.size, lon.size)
    return xr.Dataset(
        {'t': (('time', 'level', 'latitude', 'longitude'), t)},
        coords={'time': [0], 'level': [1], 'latitude': lat, 'longitude': lon},
    )


def test_crop_tropo():
    ds = _global_model()
    cropped = crop_tropo(ds, [-99.3, 19.0, -98.7, 20.0], buffer=1.0)
    # AOI plus 1 degree: lat 18-21 (descending), lon -100.3 to -97.7 -> 260-262 on the 0-360 axis.
    assert cropped.latitude.values.tolist() == [21.0, 20.0, 19.0, 18.0]
    assert cropped.longitude.values.tolist() == [260.0, 261.0, 262.0]
    # Values are a straight copy of the source nodes.
    expected = ds.sel(latitude=cropped.latitude, longitude=cropped.longitude)
    assert np.array_equal(cropped.t.values, expected.t.values)


def test_crop_tropo_across_prime_meridian():
    ds = _global_model()
    cropped = crop_tropo(ds, [-0.5, 10.0, 0.5, 11.0], buffer=1.0)
    # The west piece (359) comes first so longitude stays continuous across 0.
    assert cropped.longitude.values.tolist() == [359.0, 0.0, 1.0]
    assert cropped.latitude.values.tolist() == [12.0, 11.0, 10.0, 9.0]
    expected = ds.sel(latitude=cropped.latitude, longitude=cropped.longitude)
    assert np.array_equal(cropped.t.values, expected.t.values)
