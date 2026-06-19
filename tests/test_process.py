from pathlib import Path

from hyp3_isce3.process import get_config, get_crossmul_looks, reproject_subset


def test_get_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # get_config writes insar.yaml to cwd
    reference_path = 'REFERENCE_TEST.h5'
    secondary_path = 'SECONDARY_TEST.h5'

    reference_orbit = 'REFERENCE_ORBIT.xml'
    secondary_orbit = 'SECONDARY_ORBIT.xml'

    reference_tropo = 'REFERENCE_TROPO.nc'
    secondary_tropo = 'SECONDARY_TROPO.nc'

    tec_path = 'TEC.json'
    watermask = 'WATERMASK.vrt'

    temp_yaml = Path('temp.yaml')
    temp_yaml.write_text('partial_granule_id:')

    yaml = get_config(
        reference_path,
        secondary_path,
        reference_orbit,
        secondary_orbit,
        reference_tropo,
        secondary_tropo,
        tec_path,
        watermask,
        temp_yaml,
    )
    exists_reference = False
    exists_secondary = False
    exists_orbits = False
    exists_tropo = False
    exists_tec = False
    exists_watermask = False
    with yaml.open('r') as cfg:
        lines = cfg.readlines()
        for line in lines:
            if 'reference_rslc_file' in line:
                assert reference_path in line
                exists_reference = True
            if 'secondary_rslc_file' in line:
                assert secondary_path in line
                exists_secondary = True
            if 'water_mask_file' in line:
                assert watermask in line
                exists_watermask = True
            if 'reference_orbit_file' in line or 'secondary_orbit_file' in line:
                assert reference_orbit in line or secondary_orbit in line
                exists_orbits = True
            if 'reference_troposphere_file' in line or 'secondary_troposphere_file' in line:
                assert reference_tropo in line or secondary_tropo in line
                exists_tropo = True
            if 'tec_file' in line:
                assert tec_path in line
                exists_tec = True
    assert exists_reference and exists_secondary and exists_watermask and exists_orbits and exists_tropo and exists_tec
    assert 'partial_granule_id:' in lines[-1]


def test_get_crossmul_looks(tmp_path):
    rc = tmp_path / 'rc.yaml'
    rc.write_text(
        'runconfig:\n  groups:\n    processing:\n      crossmul:\n        range_looks: 7\n        azimuth_looks: 16\n'
    )
    assert get_crossmul_looks(rc) == (16, 7)


def test_reproject_subset():
    # AOI straddling the UTM zone 11N central meridian (117W, easting 500000).
    xmin, ymin, xmax, ymax = reproject_subset([-117.1, 46.0, -116.9, 46.2], 32611)
    assert xmin < xmax and ymin < ymax
    assert 490_000 < xmin < 500_000 < xmax < 510_000
    assert 5_000_000 < ymin < ymax < 5_200_000


def test_get_config_subset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Downloaded GUNW runconfig tail (from product_path_group on) with the geocode
    # and radar_grid_cubes blocks, plus a dem_download block that uses x/y.
    tail = (
        'product_path_group:\n'
        '    sas_output_file: output/GUNW_product.h5\n'
        'processing:\n'
        '    dem_download:\n'
        '        top_left:\n'
        '            x: 1.0\n'
        '            y: 2.0\n'
        '        bottom_right:\n'
        '            x: 3.0\n'
        '            y: 4.0\n'
        '    geocode:\n'
        '        output_epsg: 9999\n'
        '        top_left:\n'
        '            y_abs: 0.0\n'
        '            x_abs: 0.0\n'
        '        bottom_right:\n'
        '            y_abs: 0.0\n'
        '            x_abs: 0.0\n'
        '    radar_grid_cubes:\n'
        '        output_epsg: 9999\n'
        '        top_left:\n'
        '            y_abs: 0.0\n'
        '            x_abs: 0.0\n'
        '        bottom_right:\n'
        '            y_abs: 0.0\n'
        '            x_abs: 0.0\n'
        'partial_granule_id: foo\n'
    )
    Path('temp.yaml').write_text(tail)

    yaml_path = get_config(
        'REF.h5',
        'SEC.h5',
        'REFORB.xml',
        'SECORB.xml',
        'REFTROP.nc',
        'SECTROP.nc',
        'TEC.json',
        'WMASK.vrt',
        Path('temp.yaml'),
        subset_utm=(100.0, 200.0, 300.0, 400.0),
        output_epsg=32611,
    )
    text = yaml_path.read_text()
    # output_epsg pinned to the AOI zone on both geocode and radar_grid_cubes.
    assert text.count('output_epsg: 32611') == 2
    assert 'output_epsg: 9999' not in text
    # top_left = (xmin, ymax), bottom_right = (xmax, ymin), in both blocks.
    assert text.count('x_abs: 100.0') == 2 and text.count('y_abs: 400.0') == 2
    assert text.count('x_abs: 300.0') == 2 and text.count('y_abs: 200.0') == 2
    # dem_download uses x/y (not x_abs/y_abs) and is left untouched.
    assert all(s in text for s in ('x: 1.0', 'y: 2.0', 'x: 3.0', 'y: 4.0'))
