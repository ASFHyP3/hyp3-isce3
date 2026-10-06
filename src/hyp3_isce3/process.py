"""isce3 processing."""

import argparse
import contextlib
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import earthaccess
import h5py
import utm
import yaml
from hyp3lib.dem import prepare_dem_geotiff
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig
from osgeo import ogr, osr

from hyp3_isce3.constants import ANCILLARY_BUFFER_DEG, DEM_SUBSET_BUFFER_DEG
from hyp3_isce3.crop_rslc import crop_streamed, geocode_subset_box, stream_skeleton
from hyp3_isce3.search import find_rslc, search_nisar


asf.constants.INTERNAL.CMR_TIMEOUT = 90


log = logging.getLogger(__name__)


def decode_rslc_name(name: str) -> dict:
    """Decode a NISAR RSLC granule name (or file path) into its relevant fields.

    Keeps the underscore-delimited field layout in one place, e.g.
    NISAR_L1_PR_RSLC_005_172_A_008_2005_DHDH_A_20251122T024618_20251122T024652_X05007_N_F_J_001

    Args:
        name: RSLC granule name, optionally with a directory/extension (e.g. a downloaded .h5).

    Returns:
        The fields the pipeline uses, keyed by name.
    """
    # Drop any directory and file extension, then split into underscore-delimited fields.
    parts = Path(name).stem.split('_')
    return {
        'cycle': parts[4],  # acquisition cycle (used as the relative-orbit field of the product id)
        'track': parts[5],
        'direction': parts[6],  # ascending/descending flag
        'frame': parts[7],
        'start_time': datetime.strptime(parts[11], '%Y%m%dT%H%M%S'),  # acquisition start
        'end_time': datetime.strptime(parts[12], '%Y%m%dT%H%M%S'),  # acquisition end
        'common': '_'.join(parts[5:11]),  # fields a compatible reference/secondary pair must share
        'dates': '_'.join(parts[11:13]),  # the start_end datetime tokens
        'center': '_'.join(parts[13:18]),  # trailing processing-center fields
        'search_keyword': '_'.join(parts[4:8]),  # cycle_track_direction_frame, to find the matching GUNW
    }


def earlier_granule_first(g1: str, g2: str) -> tuple[str, str]:
    """Order two granules so the earlier acquisition is the reference.

    Args:
        g1: First granule.
        g2: Second granule.

    Returns:
        reference: Reference (earlier) granule.
        secondary: Secondary (later) granule.
    """
    if decode_rslc_name(g1)['start_time'] <= decode_rslc_name(g2)['start_time']:
        return g1, g2
    return g2, g1


def get_config(
    reference_path: str,
    secondary_path: str,
    reference_orbit: str,
    secondary_orbit: str,
    reference_tropo: str,
    secondary_tropo: str,
    dem_path: str,
    tec_path: str,
    watermask: str,
    template_yaml: Path,
    subset_utm: tuple[float, float, float, float] | None = None,
    output_epsg: int | None = None,
) -> Path:
    """Create a configuration file for isce3 from a production GUNW runconfig.

    The production runconfig (from :func:`download_yaml`) supplies every processing setting and
    the release metadata (CRID, product versions, DOIs). Only the fields that differ for our run
    are overwritten: input and ancillary paths, the ASF on-demand identity, the granule ID
    pattern for this pair, and (for subsets) the output grid.

    Args:
        reference_path: Path of the reference scene.
        secondary_path: Path of the secondary scene.
        reference_orbit: Path of the reference orbit.
        secondary_orbit: Path of the secondary orbit.
        reference_tropo: Path of the ECMWF file for the reference scene.
        secondary_tropo: Path of the ECMWF file for the secondary scene.
        dem_path: Path of the DEM file.
        tec_path: Path of the TEC file for the reference scene.
        watermask: Path of the water mask file.
        template_yaml: Path of the downloaded production runconfig (from :func:`download_yaml`).
        subset_utm: Optional (xmin, ymin, xmax, ymax) output box in UTM meters.
        output_epsg: EPSG code of ``subset_utm``; written into the geocode blocks so the corners and projection stay consistent.

    Returns:
        yaml_file: Path of the configuration file.
    """
    runconfig = yaml.safe_load(Path(template_yaml).read_text())
    groups = runconfig['runconfig']['groups']

    # Rebuild the input and ancillary groups from our files so none of production's job-local
    # paths can leak through; only the ancillary descriptions are carried over.
    groups['input_file_group'] = {
        'reference_rslc_file': reference_path,
        'secondary_rslc_file': secondary_path,
        **{f'qa_{p}_input_file': f'output/{p.upper()}_product.h5' for p in ('rifg', 'runw', 'gunw')},
    }
    descriptions = {
        key: value
        for key, value in groups.get('dynamic_ancillary_file_group', {}).items()
        if key.endswith('_description')
    }
    groups['dynamic_ancillary_file_group'] = {
        'dem_file': dem_path,
        'water_mask_file': watermask,
        'orbit_files': {'reference_orbit_file': reference_orbit, 'secondary_orbit_file': secondary_orbit},
        'troposphere_weather_model_files': {
            'reference_troposphere_file': reference_tropo,
            'secondary_troposphere_file': secondary_tropo,
        },
        'tec_file': tec_path,
        **descriptions,
    }
    # Output and scratch locations are ours too, relative to the working directory.
    groups['product_path_group'] = {
        'product_path': 'output',
        'scratch_path': 'scratch',
        'sas_output_file': 'output/product.h5',
        'product_counter': 1,
        'qa_output_dir': 'qa_insar',
    }

    # Keep production's CRID, product versions, and DOIs; label the product as ASF on-demand.
    executable = groups['primary_executable']
    # Write RIFG and RUNW alongside the GUNW as output/<PRODUCT>_product.h5 (with 'GUNW' alone,
    # isce3 puts them in scratch and the GUNW at sas_output_file), which process_isce3 expects.
    executable['product_type'] = 'RIFG_RUNW_GUNW'
    # isce3 records 'OD' (on-demand) as processingType 'Custom'; production's 'PR' would read 'Nominal'.
    executable['processing_type'] = 'OD'
    executable['processing_center'] = 'A'
    # isce3 fills the {...} tokens; the rest identifies this pair (reference cycle, track,
    # direction, frame, secondary cycle) and the release, as in a production GUNW name.
    reference, secondary = decode_rslc_name(reference_path), decode_rslc_name(secondary_path)
    crid, accuracy, coverage = reference['center'].split('_')[:3]
    # Fall back to the reference RSLC's release if the template carries no CRID.
    executable['composite_release_id'] = executable.get('composite_release_id') or crid
    executable['partial_granule_id'] = (
        f'NISAR_{{Level}}_OD_{{ProductType}}_{reference["cycle"]}_{reference["track"]}_'
        f'{reference["direction"]}_{reference["frame"]}_{secondary["cycle"]}_{{MODE}}_{{PO}}_'
        '{RefStartDateTime}_{RefEndDateTime}_{SecStartDateTime}_{SecEndDateTime}_'
        f'{executable["composite_release_id"]}_{accuracy}_{coverage}_A_001'
    )

    # Pin the geocode and radar_grid_cubes output grids to the AOI and projection.
    # The RSLC crop carries a margin, so this geocode box is the final crop.
    if subset_utm:
        xmin, ymin, xmax, ymax = subset_utm
        for block in ('geocode', 'radar_grid_cubes'):
            grid = groups['processing'].setdefault(block, {})
            grid['output_epsg'] = output_epsg
            grid['top_left'] = {'x_abs': xmin, 'y_abs': ymax}
            grid['bottom_right'] = {'x_abs': xmax, 'y_abs': ymin}

    yaml_file = Path('insar.yaml')
    yaml_file.write_text(yaml.safe_dump(runconfig, sort_keys=False))
    return yaml_file


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
    keyword = decode_rslc_name(reference_path)['search_keyword']
    results = search_nisar('L2', 'GUNW', granule_name=f'*{keyword}*')
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
    # Same lookup as the streaming path; an RSLC granule's only file is its .h5.
    return str(earthaccess.download([find_rslc(granule_name)])[0])


def get_orbit(scene_name: str) -> str:
    """Download orbit files.

    Args:
        scene_name: Scene name.

    Returns:
        orbit_path: Path of the orbit file.
    """
    short_name = 'NISAR_OE'
    scene = decode_rslc_name(scene_name)
    start_date, end_date = scene['start_time'], scene['end_time']
    temporal = (start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'))
    # Most to least accurate: precise, medium, near-real-time, then forecast orbit ephemeris.
    for orbit_type in ('POE', 'MOE', 'NOE', 'FOE'):
        results = earthaccess.search_data(short_name=short_name, granule_name=f'*{orbit_type}*', temporal=temporal)
        if results:
            break
    else:
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
    start_date = decode_rslc_name(scene_name)['start_time']
    day = datetime(start_date.year, start_date.month, start_date.day)
    # ECMWF weather models are 6-hourly (00/06/12/18 UTC); round to the nearest one.
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
    scene = decode_rslc_name(scene_name)
    start_date, end_date = scene['start_time'], scene['end_time']
    temporal = (start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'))
    results = earthaccess.search_data(short_name=short_name, temporal=temporal)
    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_watermask(reference_path: str, subset: list[float] | None = None) -> str:
    """Download files to apply ionospheric corrections.

    Args:
        reference_path: Path of the reference scene.
        subset: Optional AOI [lon_min, lat_min, lon_max, lat_max]; when set, the mask
            is fetched over the AOI instead of the whole frame (the 1-degree buffer
            below covers the crop margin).

    Returns:
        tropo_path: Path of the file.
    """
    short_name = 'NISAR_WATERMASK'
    if subset is None:
        poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
        bbox = poly.bounds
    else:
        bbox = subset
    buf = ANCILLARY_BUFFER_DEG
    bbox = (bbox[0] - buf, bbox[1] - buf, bbox[2] + buf, bbox[3] + buf)
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


def get_scene_polygon(reference_path: str, subset: list[float] | None = None) -> ogr.Geometry:
    """Get Polygon for reference scene.

    Args:
        reference_path: Path of the downloaded h5 file.
        subset: Optional AOI [lon_min, lat_min, lon_max, lat_max]; when set, the DEM is
            staged over the AOI (plus a buffer for the crop margin and radar-processing
            edges) instead of the whole frame. EPSG is still taken from the full scene.

    Returns:
        geom: Polygon of the reference scene.
    """
    poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
    epsg_code = get_epsg(poly.centroid.y, poly.centroid.x)
    if subset is None:
        poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg=str(epsg_code))
    else:
        # Buffer the AOI past the 512-px crop margin's ground extent (~5-6 km); the
        # extra apply_margin_to_geographic_box 5 km below then adds further headroom.
        buf = DEM_SUBSET_BUFFER_DEG
        bbox = [subset[0] - buf, subset[1] - buf, subset[2] + buf, subset[3] + buf]
        poly, _ = stage_dem.determine_polygon(reference_path, bbox=bbox, bbox_epsg='4326')
    poly = stage_dem.apply_margin_to_geographic_box(poly)
    geom = ogr.CreateGeometryFromWkt(str(poly))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    geom.AssignSpatialReference(srs)

    return geom, epsg_code


def get_product_id(reference_scene: str, secondary_scene: str) -> str:
    """Get a pre-processing name for the pair, and check the scenes are compatible.

    This names the working folder before any RSLC is read. The delivered product files are
    named by the granule ID isce3 writes into the GUNW instead (see :func:`_run_pair`), which
    follows the NISAR GUNW filename convention; this name keeps the RSLC mode/polarization
    fields and full acquisition times, so the two differ.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.

    Returns:
        product_id: Name for the pair's working folder.
    """
    level = 'L2'
    product_type = 'GUNW'
    reference = decode_rslc_name(reference_scene)
    secondary = decode_rslc_name(secondary_scene)
    if reference['common'] != secondary['common']:
        raise ValueError(f'The scenes {reference_scene} and {secondary_scene} are incompatible')
    # Relabel the producer code (field 4 of the center block) from JPL 'J' to ASF 'A',
    # targeting just that field so a 'J' elsewhere in the block is never touched.
    center_fields = reference['center'].split('_')  # e.g. ['X05007', 'N', 'F', 'J', '001']
    center_fields[3] = 'A'
    center = '_'.join(center_fields)

    product_id = (
        f'NISAR_{level}_OD_{product_type}_{reference["cycle"]}_{reference["common"]}_'
        f'{reference["dates"]}_{secondary["dates"]}_{center}'
    )

    return product_id


def process_isce3(reference_scene: str, secondary_scene: str, subset: list[float] | None = None) -> Path:
    """Run the isce3 InSAR workflow to produce a GUNW for a pair of RSLCs.

    Everything is staged and processed in a folder named for the product, so several runs
    in one working directory don't overwrite each other's files.

    Args:
        reference_scene: Name of one input scene (ordered internally against the other).
        secondary_scene: Name of the other input scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.

    Returns:
        zip_path: Absolute path of the zip holding the GUNW h5 file and its runconfig.
    """
    # Use the earlier acquisition as the reference regardless of the order passed in.
    reference_scene, secondary_scene = earlier_granule_first(reference_scene, secondary_scene)
    product_id = get_product_id(reference_scene, secondary_scene)

    run_dir = Path(product_id).resolve()
    run_dir.mkdir(exist_ok=True)
    with contextlib.chdir(run_dir):
        zip_name = _run_pair(reference_scene, secondary_scene, subset)
    return run_dir / zip_name


def _run_pair(reference_scene: str, secondary_scene: str, subset: list[float] | None) -> str:
    """Stage the inputs, run isce3, and zip the GUNW, all in the current directory.

    Args:
        reference_scene: Name of the reference (earlier) scene.
        secondary_scene: Name of the secondary (later) scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.

    Returns:
        zip_name: File name of the zip holding the GUNW h5 file and its runconfig.
    """
    earthaccess.login()
    if subset:
        # Stream a metadata-only skeleton instead of the full RSLC; it stands in for the
        # product in every pre-crop step, and the image window is streamed by crop_streamed.
        reference_path = str(stream_skeleton(reference_scene))
        secondary_path = str(stream_skeleton(secondary_scene))
    else:
        reference_path = download_rslc(reference_scene)
        secondary_path = download_rslc(secondary_scene)

    # When subsetting, stage the water mask and DEM over the AOI only (orbit/tropo/tec
    # are temporal, so they are unaffected by the subset).
    watermask = get_watermask(reference_path, subset)

    reference_orbit = get_orbit(reference_scene)
    secondary_orbit = get_orbit(secondary_scene)

    reference_tropo = get_tropo(reference_scene)
    secondary_tropo = get_tropo(secondary_scene)

    tec_path = get_tec(reference_scene)

    scene_polygon, epsg_code = get_scene_polygon(reference_path, subset)
    dem_path = get_dem(scene_polygon, epsg_code)

    # The JPL runconfig is both our config template (its tail) and the source of the
    # crossmul looks the crop aligns to; download once and reuse for both.
    template_yaml = download_yaml(reference_path)

    # Crop the RSLCs to the AOI before processing so the radar-domain steps run on a
    # small patch; the crop floors its origin to the crossmul looks (see crop_rslc).
    subset_utm = None
    if subset:
        # Reproject the AOI to the output UTM box and snap it to the geocode grid so the subset's
        # pixels coincide with a full-frame run's (else a sub-pixel offset re-rolls speckle in
        # decorrelated areas).
        subset_utm = geocode_subset_box(subset, epsg_code, template_yaml)
        az_looks, rg_looks = get_crossmul_looks(template_yaml)
        # Stream each RSLC's AOI window into a cropped <scene>.h5, windowed
        # independently from its own orbit; only overlapping image chunks are pulled.
        reference_path = crop_streamed(
            reference_scene, reference_path, subset, dem_path, az_looks=az_looks, rg_looks=rg_looks
        )
        secondary_path = crop_streamed(
            secondary_scene, secondary_path, subset, dem_path, az_looks=az_looks, rg_looks=rg_looks
        )

    yaml_path = get_config(
        reference_path,
        secondary_path,
        reference_orbit,
        secondary_orbit,
        reference_tropo,
        secondary_tropo,
        dem_path,
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
    # Name the product by the granule ID isce3 wrote into it, which follows the NISAR GUNW
    # filename convention: isce3 derives the GUNW mode and polarization codes from the RSLCs'
    # bandwidths and processed polarizations, and uses the (cropped) acquisition times.
    with h5py.File(output, 'r') as gunw:
        band = next(iter(gunw['science']))
        granule_id = gunw[f'science/{band}/identification/granuleId'][()].decode()
    output = output.rename(f'{granule_id}.h5')
    yaml_path = yaml_path.rename(f'{granule_id}.rc.yaml')

    with zipfile.ZipFile(f'{granule_id}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.write(str(output))
        zip_ref.write(str(yaml_path))
    return f'{granule_id}.zip'
