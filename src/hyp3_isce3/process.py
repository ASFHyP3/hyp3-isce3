"""isce3 processing."""

import argparse
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import earthaccess
import utm
from hyp3lib.dem import prepare_dem_geotiff
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig
from osgeo import ogr, osr

import hyp3_isce3


log = logging.getLogger(__name__)


def get_config(
    reference_path: str,
    secondary_path: str,
    reference_orbit: str,
    secondary_orbit: str,
    reference_tropo: str,
    secondary_tropo: str,
    tec_path: str,
    watermask: str,
) -> Path:
    """Create a configuration file for isce3.

    Args:
        reference_path: Path of the reference scene.
        secondary_path: Path of the secondary scene.
        reference_orbit: Path of the reference orbit.
        secondary_orbit: Path of the secondary orbit.
        reference_tropo: Path of the ECMWF file for the reference scene.
        secondary_tropo: Path of the ECMWF file for the secondary scene.
        tec_path: Path of the TEC file for the reference scene.
        watermask: Path of the water mask file.

    Returns:
        yaml_file: Path of the configuration file.
    """
    tmp_yaml = download_yaml(reference_path)
    with tmp_yaml.open('r') as yaml:
        lines_tmp = yaml.readlines()
        for first, line in enumerate(lines_tmp):
            if 'product_path_group:' in line:
                break
    tmp_yaml.unlink()

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
            elif 'reference_orbit' in line:
                newstring += line.replace('ref_orbit', reference_orbit)
            elif 'secondary_orbit' in line:
                newstring += line.replace('sec_orbit', secondary_orbit)
            elif 'reference_tropo' in line:
                newstring += line.replace('ref_tropo', reference_tropo)
            elif 'secondary_tropo' in line:
                newstring += line.replace('sec_tropo', secondary_tropo)
            elif 'tec_file' in line:
                newstring += line.replace('tec_path', tec_path)
            elif 'watermask' in line:
                newstring += line.replace('watermask', watermask)
            else:
                newstring = line
            yaml.write(newstring)

        for line in lines_tmp[first::]:
            yaml.write(line)

    return Path('insar.yaml')


def download_yaml(reference_path: str) -> Path:
    """Download reference configuration file for GUNW.

    Args:
        reference_path: Path of the reference scene.

    Returns:
        tmp_path: Path of the yaml file.
    """
    short_name = 'NISAR_L2_GUNW_BETA_V1'
    keyword = '_'.join(reference_path.split('_')[4:8])
    results = earthaccess.search_data(short_name=short_name, granule_name=f'*{keyword}*')
    gunw = results[0].data_links()[0].split('/')[-2]
    res = asf.granule_search(gunw)
    yaml_url = res.find_urls(pattern=r'.yaml')[0]
    asf.download_url(url=yaml_url, path='./', filename='temp.yaml')

    tmp_yaml = Path('temp.yaml')

    return tmp_yaml


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

    for scene in Path().glob(f'{granule_name}*.h5'):
        h5file_path = str(scene)
    return h5file_path


def get_orbit(scene_name: str) -> str:
    """Download orbit files.

    Args:
        scene_name: Scene name.

    Returns:
        orbit_path: Path of the orbit file.
    """
    short_name = 'NISAR_OE'
    start_date = datetime.strptime(scene_name.split('_')[11], '%Y%m%dT%H%M%S')
    end_date = datetime.strptime(scene_name.split('_')[12], '%Y%m%dT%H%M%S')
    temporal = (start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'))
    results = earthaccess.search_data(short_name=short_name, granule_name='*POE*', temporal=temporal)
    if len(results) == 0:
        results = earthaccess.search_data(short_name=short_name, granule_name='*MOE*', temporal=temporal)
    if len(results) == 0:
        results = earthaccess.search_data(short_name=short_name, granule_name='*NOE*', temporal=temporal)
    if len(results) == 0:
        results = earthaccess.search_data(short_name=short_name, granule_name='*FOE*', temporal=temporal)
    if len(results) == 0:
        raise RuntimeError(f'Orbit for scene {scene_name} not found')

    files = sorted(earthaccess.download(results))
    return str(files[-1])


def get_tropo(scene_name: str) -> str:
    """Download files to apply tropospheric corrections.

    Args:
        scene_name: Scene name.

    Returns:
        tropo_path: Path of the file.
    """
    short_name = 'ASF_ECMWF_TROP'
    start_date = datetime.strptime(scene_name.split('_')[11], '%Y%m%dT%H%M%S')
    day = datetime(start_date.year, start_date.month, start_date.day)
    if start_date.hour % 6 < 3:
        tropo_date = day + timedelta(hours=int(start_date.hour / 6) * 6)
    else:
        tropo_date = day + timedelta(hours=int(start_date.hour / 6 + 1) * 6)

    temporal = (tropo_date.strftime('%Y-%m-%d %H'), tropo_date.strftime('%Y-%m-%d %H'))
    results = earthaccess.search_data(short_name=short_name, temporal=temporal)

    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_tec(scene_name: str) -> str:
    """Download files to apply ionospheric corrections.

    Args:
        scene_name: Scene name.

    Returns:
        tropo_path: Path of the file.
    """
    short_name = 'NISAR_TEC'
    start_date = datetime.strptime(scene_name.split('_')[11], '%Y%m%dT%H%M%S')
    end_date = datetime.strptime(scene_name.split('_')[12], '%Y%m%dT%H%M%S')
    temporal = (start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'))
    results = earthaccess.search_data(short_name=short_name, temporal=temporal)
    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_watermask(reference_path: str) -> str:
    """Download files to apply ionospheric corrections.

    Args:
        reference_path: Path of the reference scene.

    Returns:
        tropo_path: Path of the file.
    """
    short_name = 'NISAR_WATERMASK'
    poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
    bbox = poly.bounds
    bbox = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
    results = earthaccess.search_data(short_name=short_name, bounding_box=bbox)
    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_dem(scene_poly: ogr.Geometry, epsg_code: int, dem_path: str = 'dem.tif') -> str:
    """Download DEM for a given polygon.

    Args:
        scene_poly: Scene polygon.
        epsg_code: EPSG code for the output projection.
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


def get_epsg(lat: float, lon: float) -> int:
    """Get EPSG code from Polygon.

    Args:
        lat: Latitude coordinate of the centroid.
        lon: Longitude coordinate of the centroid.

    Returns:
        epsg_code: EPSG code for the polygon projection.
    """
    _, _, zone_number, zone_letter = utm.from_latlon(lat, lon)

    is_northern = zone_letter >= 'N'

    epsg_base = 32600 if is_northern else 32700
    return epsg_base + zone_number


def get_scene_polygon(reference_path: str) -> ogr.Geometry:
    """Get Polygon for reference scene.

    Args:
        reference_path: Path of the downloaded h5 file.
        epsg_code: EPSG code for the polygon coordinates.

    Returns:
        geom: Polygon of the reference scene.
    """
    poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
    epsg_code = get_epsg(poly.centroid.y, poly.centroid.x)
    poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg=str(epsg_code))
    poly = stage_dem.apply_margin_to_geographic_box(poly)
    geom = ogr.CreateGeometryFromWkt(str(poly))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    geom.AssignSpatialReference(srs)

    return geom, epsg_code


def get_product_id(reference_scene: str, secondary_scene: str) -> str:
    """Get product name for GUNW.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.

    Returns:
        product_id: Name for the GUNW.
    """
    level = 'L2'
    product_type = 'GUNW'
    rel_orb = reference_scene.split('_')[4]
    common = '_'.join(reference_scene.split('_')[5:11])
    if not common == '_'.join(secondary_scene.split('_')[5:11]):
        raise ValueError(f'The scenes {reference_scene} and {secondary_scene} are incompatible')
    ref_dates = '_'.join(reference_scene.split('_')[11:13])
    sec_dates = '_'.join(secondary_scene.split('_')[11:13])
    center = '_'.join(reference_scene.split('_')[13:18]).replace('J', 'A')

    product_id = f'NISAR_{level}_OD_{product_type}_{rel_orb}_{common}_{ref_dates}_{sec_dates}_{center}'

    return product_id


def process_isce3(reference_scene: str, secondary_scene: str) -> Path:
    """Get Polygon for reference scene.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.

    Returns:
        h5file: Path of the GUNW h5file.
    """
    product_id = get_product_id(reference_scene, secondary_scene)

    earthaccess.login()
    reference_path = download_rslc(reference_scene)
    secondary_path = download_rslc(secondary_scene)

    watermask = get_watermask(reference_path)

    reference_orbit = get_orbit(reference_scene)
    secondary_orbit = get_orbit(secondary_scene)

    reference_tropo = get_tropo(reference_scene)
    secondary_tropo = get_tropo(secondary_scene)

    tec_path = get_tec(reference_scene)

    scene_polygon, epsg_code = get_scene_polygon(reference_path)
    _ = get_dem(scene_polygon, epsg_code)
    yaml_path = get_config(
        reference_path,
        secondary_path,
        reference_orbit,
        secondary_orbit,
        reference_tropo,
        secondary_tropo,
        tec_path,
        watermask,
    )
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

    output = Path('output/GUNW_product.h5')
    if not output.exists():
        raise RuntimeError('The GUNW file was not written!')
    output.rename(f'{product_id}.h5')
    output = Path(f'{product_id}.h5')
    yaml_path.rename(f'{product_id}.rc.yaml')
    yaml_path = Path(f'{product_id}.rc.yaml')

    with zipfile.ZipFile(f'{product_id}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.write(str(output))
        zip_ref.write(str(yaml_path))

    return Path(f'{product_id}.zip')
