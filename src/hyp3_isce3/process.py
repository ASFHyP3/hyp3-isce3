"""isce3 processing."""

import argparse
import logging
from pathlib import Path

import asf_search as asf
from hyp3lib.dem import prepare_dem_geotiff
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig
from osgeo import ogr, osr

import hyp3_isce3


log = logging.getLogger(__name__)


def get_config(reference_path: str, secondary_path: str) -> Path:
    """Create a configuration file for isce3.

    Args:
        reference_path: Path of the reference scene.
        secondary_path: Path of the secondary scene.

    Returns:
        yaml_file: Path of the configuration file.
    """
    yaml_schema = Path(hyp3_isce3.__file__).parent / 'schemas' / 'insar.yaml'
    with yaml_schema.open('r') as yaml:
        lines = yaml.readlines()

    yaml_file = Path('insar.yaml')
    with yaml_file.open('w') as yaml:
        for line in lines:
            newstring = ''
            if 'reference_scene' in line:
                newstring += line.replace('reference_scene', reference_path)
            elif 'secondary_scene' in line:
                newstring += line.replace('secondary_scene', secondary_path)
            else:
                newstring = line
            yaml.write(newstring)

    return Path('insar.yaml')


def download_rslc(granule_name: str) -> str:
    """Download RSLC product.

    Args:
        granule_name: Name of the scene.

    Returns:
        h5file_path: Path of the h5 file.
    """
    res = asf.granule_search([granule_name])

    if len(res) == 0:
        raise ValueError(f'`asf_search` was unable to find {granule_name}')

    res.download(path='.')

    for scene in Path().glob(f"{granule_name}*.h5"):
        h5file_path = str(scene)
    return h5file_path


def get_dem(scene_poly: ogr.Geometry, dem_path: str = 'dem.tif') -> str:
    """Download DEM for a given polygon.

    Args:
        scene_poly: Scene polygon.
        dem_path: Output path for the DEM.

    Returns:
        dem_path: Path of the DEM file.
    """
    return str(
        prepare_dem_geotiff(
            output_name=dem_path,
            geometry=scene_poly,
            epsg_code=4326,
            pixel_size=0.001,
        )
    )


def get_scene_polygon(reference_path: str, epsg_code: int = 4326) -> ogr.Geometry:
    """Get Polygon for reference scene.

    Args:
        reference_path: Path of the downloaded h5 file.
        epsg_code: EPSG code for the polygon coordinates.

    Returns:
        geom: Polygon of the reference scene.
    """
    poly, epsg = stage_dem.determine_polygon(reference_path, bbox = None, bbox_epsg = epsg_code)
    poly = stage_dem.apply_margin_to_geographic_box(poly)
    geom = ogr.CreateGeometryFromWkt(str(poly))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    geom.AssignSpatialReference(srs)

    return geom


def process_isce3(reference_scene: str, secondary_scene: str) -> Path:
    """Get Polygon for reference scene.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.

    Returns:
        h5file: Path of the GUNW h5file.
    """
    reference_path = download_rslc(reference_scene)
    secondary_path = download_rslc(secondary_scene)
    
    scene_polygon = get_scene_polygon(reference_path)
    get_dem(scene_polygon)
    yaml_path = get_config(reference_path, secondary_path)
    args = argparse.Namespace(run_config_path=str(yaml_path), log_file=False)
    insar_runcfg = InsarRunConfig(args)

    run_steps = {
        'bandpass_insar': True,
        'rdr2geo': True,
        'geo2rdr': True,
        'prepare_insar_hdf5': True,
        'coarse_resample': True,
        'dense_offsets': True,
        'offsets_product': True,
        'rubbersheet': True,
        'fine_resample': True,
        'crossmul': True,
        'filter_interferogram': True,
        'unwrap': True,
        'ionosphere': True,
        'geocode': True,
        'solid_earth_tides': True,
        'baseline': True,
        'troposphere': True,
    }

    _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

    insar.run(cfg=insar_runcfg.cfg, out_paths=out_paths, run_steps=run_steps)
    
    if not Path('GUNW.h5').exists():
        raise RuntimeError('The GUNW file was not written!')
    return Path('GUNW.h5')
