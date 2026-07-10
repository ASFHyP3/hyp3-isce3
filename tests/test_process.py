from pathlib import Path

import pytest
import yaml

from hyp3_isce3.process import _staged, apply_overrides, get_config, get_crossmul_looks


def _runconfig(tmp_path: Path) -> Path:
    rc = tmp_path / 'rc.yaml'
    rc.write_text(
        'runconfig:\n'
        '  groups:\n'
        '    processing:\n'
        '      crossmul:\n'
        '        range_looks: 7\n'
        '        azimuth_looks: 16\n'
        '      dense_offsets:\n'
        '        enabled: true\n'
    )
    return rc


def test_apply_overrides_nested_and_dotted(tmp_path):
    rc = _runconfig(tmp_path)
    # A dotted key and a nested dict both under `processing` -- both must survive the merge.
    apply_overrides(
        rc,
        {
            'processing.crossmul.range_looks': 11,
            'processing': {'dense_offsets': {'enabled': False}},
        },
    )
    proc = yaml.safe_load(rc.read_text())['runconfig']['groups']['processing']
    assert proc['crossmul']['range_looks'] == 11
    assert proc['crossmul']['azimuth_looks'] == 16  # untouched key preserved
    assert proc['dense_offsets']['enabled'] is False


def test_apply_overrides_unknown_key_raises(tmp_path):
    rc = _runconfig(tmp_path)
    with pytest.raises(KeyError, match='does not exist'):
        apply_overrides(rc, {'processing.crossmul.rnge_looks': 9})


def test_apply_overrides_leaf_as_section_raises(tmp_path):
    rc = _runconfig(tmp_path)
    with pytest.raises(KeyError, match='leaf'):
        apply_overrides(rc, {'processing.crossmul.range_looks': {'nope': 1}})


def test_staged_no_cache_dir_is_passthrough(tmp_path):
    src = tmp_path / 'produced.bin'
    src.write_text('x')
    calls = []

    def produce():
        calls.append(1)
        return str(src)

    # With cache_dir=None, _staged just returns produce()'s path every time.
    assert _staged(None, 'name.bin', produce) == str(src)
    assert _staged(None, 'name.bin', produce) == str(src)
    assert len(calls) == 2


def test_staged_caches_and_reuses(tmp_path):
    cache = tmp_path / 'cache'
    calls = []

    def produce():
        calls.append(1)
        # produce() writes its own file somewhere; _staged moves it into the cache.
        out = tmp_path / 'fresh.bin'
        out.write_text('data')
        return str(out)

    # First call: cache miss -> produce runs, result stashed under the deterministic name.
    first = _staged(str(cache), 'orbit.xml', produce)
    assert first == str(cache / 'orbit.xml')
    assert (cache / 'orbit.xml').read_text() == 'data'
    assert not (tmp_path / 'fresh.bin').exists()  # moved, not copied

    # Second call: cache hit -> produce is NOT run, cached path returned.
    second = _staged(str(cache), 'orbit.xml', produce)
    assert second == str(cache / 'orbit.xml')
    assert len(calls) == 1


def test_staged_already_at_target_not_moved(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    target = cache / 'dem.tif'
    target.write_text('dem')

    # produce() returns a path that is already the cache target (e.g. a pre-placed crop) --
    # _staged must not try to move a file onto itself.
    result = _staged(str(cache), 'dem.tif', lambda: str(target))
    assert result == str(target)
    assert target.read_text() == 'dem'


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
