from pathlib import Path

import yaml

from hyp3_isce3.process import get_config, get_crossmul_looks


REFERENCE = 'NISAR_L1_PR_RSLC_005_019_A_011_4005_DHDH_A_20251111T120539_20251111T120614_P05023_N_F_J_001'
SECONDARY = 'NISAR_L1_PR_RSLC_006_019_A_011_4005_DHDH_A_20251123T120540_20251123T120614_P05023_N_F_J_001'


def _write_template(path: Path) -> Path:
    """Write a production-like GUNW runconfig with job-local paths and production identity."""
    template = {
        'runconfig': {
            'name': 'NISAR_INSAR_RUNCONFIG',
            'groups': {
                'pge_name_group': {'pge_name': 'INSAR_L_PGE'},
                'input_file_group': {
                    'reference_rslc_file': '/data/work/ref.h5',
                    'secondary_rslc_file': '/data/work/sec.h5',
                    'qa_gunw_input_file': 'output/product.h5',
                },
                'dynamic_ancillary_file_group': {
                    'dem_file': '/data/work/dem.vrt',
                    'dem_file_description': 'NISAR DEM',
                    'water_mask_file': '/data/work/wm.vrt',
                    'orbit_files': {'reference_orbit_file': '/data/a.xml', 'secondary_orbit_file': '/data/b.xml'},
                    'troposphere_weather_model_files': {
                        'reference_troposphere_file': '/data/a.nc',
                        'secondary_troposphere_file': '/data/b.nc',
                    },
                    'tec_file': '/data/tec.json',
                    'static_layers_file': '/data/work/static.h5',
                },
                'primary_executable': {
                    'product_type': 'GUNW',
                    'product_version': {'gunw_version': '1.0.11'},
                    'product_doi': {'gunw_doi': '10.5067/NIL2GUNW-P1'},
                    'composite_release_id': 'P05023',
                    'processing_type': 'PR',
                    'processing_center': 'J',
                    'partial_granule_id': 'production pattern',
                },
                'product_path_group': {'product_path': '/data/work/output', 'sas_output_file': '/data/work/out.h5'},
                'processing': {
                    'dem_download': {'top_left': {'x': None, 'y': None}, 'bottom_right': {'x': None, 'y': None}},
                    'geocode': {
                        'output_epsg': 9999,
                        'top_left': {'y_abs': 0.0, 'x_abs': 0.0},
                        'bottom_right': {'y_abs': 0.0, 'x_abs': 0.0},
                    },
                    'radar_grid_cubes': {
                        'output_epsg': 9999,
                        'top_left': {'y_abs': None, 'x_abs': None},
                        'bottom_right': {'y_abs': None, 'x_abs': None},
                    },
                    'crossmul': {'range_looks': 5, 'azimuth_looks': 6},
                },
            },
        }
    }
    path.write_text(yaml.safe_dump(template, sort_keys=False))
    return path


def _get_config(
    tmp_path: Path,
    subset_utm: tuple[float, float, float, float] | None = None,
    output_epsg: int | None = None,
) -> dict:
    template = _write_template(tmp_path / 'temp.yaml')
    yaml_path = get_config(
        f'{REFERENCE}.h5',
        f'{SECONDARY}.h5',
        'REFORB.xml',
        'SECORB.xml',
        'REFTROP.nc',
        'SECTROP.nc',
        'DEM.tif',
        'TEC.json',
        'WMASK.vrt',
        template,
        subset_utm=subset_utm,
        output_epsg=output_epsg,
    )
    return yaml.safe_load(yaml_path.read_text())['runconfig']['groups']


def test_get_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # get_config writes insar.yaml to cwd
    groups = _get_config(tmp_path)

    # Our inputs and ancillary files replace production's job-local paths.
    inputs = groups['input_file_group']
    assert inputs['reference_rslc_file'] == f'{REFERENCE}.h5'
    assert inputs['secondary_rslc_file'] == f'{SECONDARY}.h5'
    assert inputs['qa_gunw_input_file'] == 'output/GUNW_product.h5'
    ancillary = groups['dynamic_ancillary_file_group']
    assert ancillary['dem_file'] == 'DEM.tif'
    assert ancillary['water_mask_file'] == 'WMASK.vrt'
    assert ancillary['tec_file'] == 'TEC.json'
    assert ancillary['orbit_files'] == {'reference_orbit_file': 'REFORB.xml', 'secondary_orbit_file': 'SECORB.xml'}
    assert ancillary['troposphere_weather_model_files'] == {
        'reference_troposphere_file': 'REFTROP.nc',
        'secondary_troposphere_file': 'SECTROP.nc',
    }
    # Descriptions are carried over; production-only file paths are dropped.
    assert ancillary['dem_file_description'] == 'NISAR DEM'
    assert 'static_layers_file' not in ancillary
    # Output locations are ours, not production's job directory.
    assert groups['product_path_group']['product_path'] == 'output'
    assert groups['product_path_group']['sas_output_file'] == 'output/product.h5'

    # Production's release metadata is kept; the product is labeled ASF on-demand.
    executable = groups['primary_executable']
    assert executable['composite_release_id'] == 'P05023'
    assert executable['product_version'] == {'gunw_version': '1.0.11'}
    assert executable['product_doi'] == {'gunw_doi': '10.5067/NIL2GUNW-P1'}
    assert executable['product_type'] == 'RIFG_RUNW_GUNW'
    assert executable['processing_type'] == 'OD'
    assert executable['processing_center'] == 'A'
    assert executable['partial_granule_id'] == (
        'NISAR_{Level}_OD_{ProductType}_005_019_A_011_006_{MODE}_{PO}_'
        '{RefStartDateTime}_{RefEndDateTime}_{SecStartDateTime}_{SecEndDateTime}_P05023_N_F_A_001'
    )

    # Without a subset the processing settings pass through untouched.
    assert groups['processing']['geocode']['output_epsg'] == 9999
    assert groups['processing']['crossmul'] == {'range_looks': 5, 'azimuth_looks': 6}


def test_get_crossmul_looks(tmp_path):
    rc = tmp_path / 'rc.yaml'
    rc.write_text(
        'runconfig:\n  groups:\n    processing:\n      crossmul:\n        range_looks: 7\n        azimuth_looks: 16\n'
    )
    assert get_crossmul_looks(rc) == (16, 7)


def test_get_config_without_crid(monkeypatch, tmp_path):
    # A template without a CRID falls back to the reference RSLC's release.
    monkeypatch.chdir(tmp_path)
    template = _write_template(tmp_path / 'temp.yaml')
    runconfig = yaml.safe_load(template.read_text())
    del runconfig['runconfig']['groups']['primary_executable']['composite_release_id']
    template.write_text(yaml.safe_dump(runconfig))
    yaml_path = get_config(
        f'{REFERENCE}.h5',
        f'{SECONDARY}.h5',
        'REFORB.xml',
        'SECORB.xml',
        'REFTROP.nc',
        'SECTROP.nc',
        'DEM.tif',
        'TEC.json',
        'WMASK.vrt',
        template,
    )
    executable = yaml.safe_load(yaml_path.read_text())['runconfig']['groups']['primary_executable']
    assert executable['composite_release_id'] == 'P05023'
    assert executable['partial_granule_id'].endswith('_P05023_N_F_A_001')


def test_get_config_subset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    groups = _get_config(tmp_path, subset_utm=(100.0, 200.0, 300.0, 400.0), output_epsg=32611)

    # output_epsg and corners pinned to the AOI on both geocode and radar_grid_cubes:
    # top_left = (xmin, ymax), bottom_right = (xmax, ymin).
    for block in ('geocode', 'radar_grid_cubes'):
        grid = groups['processing'][block]
        assert grid['output_epsg'] == 32611
        assert grid['top_left'] == {'x_abs': 100.0, 'y_abs': 400.0}
        assert grid['bottom_right'] == {'x_abs': 300.0, 'y_abs': 200.0}
    # dem_download uses x/y (not x_abs/y_abs) and is left untouched.
    assert groups['processing']['dem_download']['top_left'] == {'x': None, 'y': None}
