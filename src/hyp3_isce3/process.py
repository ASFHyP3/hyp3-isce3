"""isce3 processing."""

import argparse
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import earthaccess
import utm
import yaml
from hyp3lib.dem import prepare_dem_geotiff
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig
from osgeo import ogr, osr
from pyproj import Transformer

import hyp3_isce3
from hyp3_isce3.crop_rslc import crop_rslc_pair


asf.constants.INTERNAL.CMR_TIMEOUT = 90


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
    template_yaml: Path,
    subset_utm: tuple[float, float, float, float] | None = None,
    output_epsg: int | None = None,
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
        template_yaml: Path of the downloaded JPL runconfig (from :func:`download_yaml`) used as the tail/template.
        subset_utm: Optional (xmin, ymin, xmax, ymax) output box in UTM meters.
        output_epsg: EPSG code of ``subset_utm``; written into the geocode blocks so the corners and projection stay consistent.

    Returns:
        yaml_file: Path of the configuration file.
    """
    # Keep the downloaded runconfig's tail (product_path_group on) and splice our
    # schema header in front. The caller owns template_yaml, so we don't delete it.
    with Path(template_yaml).open('r') as f:
        lines_tmp = f.readlines()
        for first, line in enumerate(lines_tmp):
            if 'product_path_group:' in line:
                break

    # Our shipped schema is the header (input/ancillary file groups) with
    # placeholder tokens standing in for the real scene/orbit/ancillary paths.
    yaml_schema = Path(hyp3_isce3.__file__).parent / 'schemas' / 'insar.yaml'
    with yaml_schema.open('r') as f:
        lines = f.readlines()

    yaml_file = Path('insar.yaml')
    with yaml_file.open('w') as out:
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
            out.write(newstring)

        # Pin the geocode and radar_grid_cubes output grids to the AOI and projection.
        # The RSLC crop carries a margin, so this geocode box is the final crop.
        xmin, ymin, xmax, ymax = subset_utm if subset_utm else (None, None, None, None)
        corner = 0
        for line in lines_tmp[first::]:
            if subset_utm is None:
                pass
            elif 'output_epsg' in line:
                line = f'{line.partition(":")[0]}: {output_epsg}\n'
            elif 'top_left' in line:
                corner = 1
            elif 'bottom_right' in line:
                corner = 2
            elif corner and 'x_abs' in line:
                line = f'{line.partition(":")[0]}: {xmin if corner == 1 else xmax}\n'
            elif corner and 'y_abs' in line:
                line = f'{line.partition(":")[0]}: {ymax if corner == 1 else ymin}\n'
            out.write(line)

    return Path('insar.yaml')


def get_crossmul_looks(template_yaml: Path) -> tuple[int, int]:
    """Read the InSAR crossmul (azimuth, range) multilook looks from the runconfig template.

    The RSLC crop floors its window origin to these so the crop's multilook cells
    coincide with a full-frame run. Sourcing them from the same runconfig the
    workflow runs guarantees the two never drift. If looks are ever exposed as a
    user/processing parameter, read that parameter here instead so the crop tracks it.

    Args:
        template_yaml: Path of the downloaded JPL runconfig (from :func:`download_yaml`).

    Returns:
        looks: (azimuth_looks, range_looks) from the runconfig ``crossmul`` block.
    """
    crossmul = yaml.safe_load(Path(template_yaml).read_text())['runconfig']['groups']['processing']['crossmul']
    return int(crossmul['azimuth_looks']), int(crossmul['range_looks'])


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


def reproject_subset(subset: list[float], epsg_code: int) -> tuple[float, float, float, float]:
    """Reproject a WGS84 lon/lat subset box to the output UTM projection.

    Args:
        subset: WGS84 bounding box as [lon_min, lat_min, lon_max, lat_max].
        epsg_code: Output UTM EPSG code.

    Returns:
        bbox: Enclosing (xmin, ymin, xmax, ymax) box in UTM meters.
    """
    lon_min, lat_min, lon_max, lat_max = subset
    # transform_bounds densifies the edges, so the returned box encloses the
    # reprojected lon/lat rectangle even where its boundary bulges between corners.
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_code}', always_xy=True)
    return transformer.transform_bounds(lon_min, lat_min, lon_max, lat_max)


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


def process_isce3(reference_scene: str, secondary_scene: str, subset: list[float] | None = None) -> Path:
    """Get Polygon for reference scene.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.

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
    dem_path = get_dem(scene_polygon, epsg_code)

    # The JPL runconfig is both our config template (its tail) and the source of the
    # crossmul looks the crop aligns to; download once and reuse for both.
    template_yaml = download_yaml(reference_path)

    # Crop the RSLCs to the AOI before processing so the radar-domain steps run on a
    # small patch; the crop floors its origin to the crossmul looks (see crop_rslc).
    subset_utm = None
    if subset:
        subset_utm = reproject_subset(subset, epsg_code)
        az_looks, rg_looks = get_crossmul_looks(template_yaml)
        reference_path, secondary_path = crop_rslc_pair(
            reference_path,
            secondary_path,
            subset,
            dem_path,
            out_dir='.',
            margin=512,
            az_looks=az_looks,
            rg_looks=rg_looks,
        )

    yaml_path = get_config(
        reference_path,
        secondary_path,
        reference_orbit,
        secondary_orbit,
        reference_tropo,
        secondary_tropo,
        tec_path,
        watermask,
        template_yaml,
        subset_utm,
        epsg_code,
    )
    template_yaml.unlink()  # consumed by the looks read and the config build

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
