"""isce3 processing."""

import argparse
import contextlib
import logging
import re
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asf_search as asf
import earthaccess
import h5py
import utm
import xarray as xr
import yaml
from hyp3lib.aws import upload_file_to_s3
from nisar.products.readers import SLC
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig

import hyp3_isce3
from hyp3_isce3 import isce3_fast
from hyp3_isce3.cli import nullable_subset_list, parse_subset
from hyp3_isce3.crop_rslc import (
    crop_streamed,
    geocode_subset_box,
    get_multilook_looks,
    get_polarizations,
    get_processed_polarizations,
    stream_skeleton,
)
from hyp3_isce3.crop_tropo import crop_tropo
from hyp3_isce3.dem import mosaic_tiles
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
        'start_time': datetime.strptime(parts[11], '%Y%m%dT%H%M%S').replace(tzinfo=UTC),  # acquisition start
        'end_time': datetime.strptime(parts[12], '%Y%m%dT%H%M%S').replace(tzinfo=UTC),  # acquisition end
        'common': '_'.join(parts[5:11]),  # fields a compatible reference/secondary pair must share
        'dates': '_'.join(parts[11:13]),  # the start_end datetime tokens
        'center': '_'.join(parts[13:18]),  # trailing processing-center fields
        'search_keyword': '_'.join(parts[5:8]),  # track_direction_frame, to find a GUNW on the same frame
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
            Without a subset it fills only a blank output_epsg (as in the fallback runconfig).

    Returns:
        yaml_file: Path of the configuration file.
    """
    runconfig = yaml.safe_load(Path(template_yaml).read_text())
    groups = runconfig['runconfig']['groups']

    # Production runconfigs (isce3 0.25.16) write the unwrap mask_type as one value; newer isce3
    # schemas take a list. Wrap it only when the installed schema asks for a list.
    mask = groups.get('processing', {}).get('phase_unwrap', {}).get('preprocess_wrapped_phase', {}).get('mask', {})
    schema = (Path(insar.__file__).parent / 'schemas' / 'insar.yaml').read_text()
    if isinstance(mask.get('mask_type'), str) and re.search(r'^\s+mask_type: list\(', schema, re.MULTILINE):
        mask['mask_type'] = [mask['mask_type']]

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
    # A subset covers only part of the frame, so mark it partial ('P') per the NISAR naming convention.
    if subset_utm:
        coverage = 'P'
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
    elif groups['processing'].get('geocode', {}).get('output_epsg') is None:
        # Only the fallback runconfig has a blank geocode projection, and isce3 would otherwise take
        # the DEM's lat/lon EPSG; use the scene's UTM zone (isce3 copies it to radar_grid_cubes).
        groups['processing'].setdefault('geocode', {})['output_epsg'] = output_epsg

    yaml_file = Path('insar.yaml')
    yaml_file.write_text(yaml.safe_dump(runconfig, sort_keys=False))
    return yaml_file


def download_yaml(keyword: str) -> Path:
    """Download the production GUNW runconfig for a frame, or copy the fallback when there is none.

    The subset focus calls this too, so it snaps to the same multilook looks this workflow will use.

    Args:
        keyword: track_direction_frame of the frame, e.g. '085_D_069'.

    Returns:
        tmp_yaml: Path of the yaml file.
    """
    try:
        results = search_nisar('L2', 'GUNW', granule_name=f'*{keyword}*')
    except ValueError:
        # No production GUNW on this frame yet; use the shipped runconfig instead. Copy it,
        # since the caller deletes temp.yaml once it is consumed.
        log.warning(f'No production GUNW found for frame {keyword}; using the fallback runconfig.')
        fallback = Path(hyp3_isce3.__file__).parent / 'schemas' / 'gunw_fallback.yaml'
        return Path(shutil.copy(fallback, 'temp.yaml'))
    gunw = results[0].data_links()[0].split('/')[-2]
    res = asf.granule_search(gunw)
    yaml_url = res.find_urls(pattern=r'.yaml')[0]
    asf.download_url(url=yaml_url, path='./', filename='temp.yaml')

    tmp_yaml = Path('temp.yaml')

    return tmp_yaml


def download_rslc(granule_name: str) -> str:
    """Download RSLC product, or use it as is when it is already a local file.

    Args:
        granule_name: Name of the scene, or the path of a local RSLC h5 file.

    Returns:
        h5file_path: Path of the h5 file.
    """
    # A local RSLC (e.g. one focused with ++process focus_l0b) is not in CMR.
    if Path(granule_name).is_file():
        return granule_name
    # Same lookup as the streaming path; keep only the .h5 in case the granule ships other files.
    files = earthaccess.download([find_rslc(granule_name)])
    h5_files = [f for f in files if str(f).endswith('.h5')]
    if not h5_files:
        raise RuntimeError(f'No .h5 file downloaded for {granule_name}')
    return str(h5_files[0])


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


def get_tropo(scene_name: str, subset: list[float] | None = None) -> str:
    """Download files to apply tropospheric corrections.

    Args:
        scene_name: Scene name.
        subset: Optional AOI [lon_min, lat_min, lon_max, lat_max]; when set, only the
            AOI (plus a 1-degree buffer) of the global weather model is streamed
            instead of downloading the whole ~2 GB file.

    Returns:
        tropo_path: Path of the file.
    """
    short_name = 'ASF_ECMWF_TROP'
    start_date = decode_rslc_name(scene_name)['start_time']
    day = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
    # ECMWF weather models are 6-hourly (00/06/12/18 UTC); round to the nearest one.
    if start_date.hour % 6 < 3:
        tropo_date = day + timedelta(hours=int(start_date.hour / 6) * 6)
    else:
        tropo_date = day + timedelta(hours=int(start_date.hour / 6 + 1) * 6)

    temporal = (tropo_date.strftime('%Y-%m-%d %H'), tropo_date.strftime('%Y-%m-%d %H'))
    results = earthaccess.search_data(short_name=short_name, temporal=temporal)

    if subset:
        # Stream only the chunks overlapping the AOI and write them to a small local file.
        tropo_path = Path(f'{results[-1]["meta"]["native-id"]}_subset.nc')
        fileobj = earthaccess.open(results[-1:])[0]
        with xr.open_dataset(fileobj, engine='h5netcdf') as ds:
            cropped = crop_tropo(ds, subset).load()
        fileobj.close()
        for var in cropped.variables.values():
            # Keep the source encoding (fill values, compression, time units) but drop its
            # chunk sizes, which are larger than the crop.
            var.encoding.pop('chunksizes', None)
        cropped.to_netcdf(tropo_path)
        log.info(f'Wrote AOI weather model {tropo_path} with shape {dict(cropped.sizes)}')
        return str(tropo_path)

    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_tec(scene_name: str) -> str:
    """Download files to apply ionospheric corrections.

    Args:
        scene_name: Scene name.

    Returns:
        tec_path: Path of the TEC file.
    """
    short_name = 'NISAR_TEC'
    scene = decode_rslc_name(scene_name)
    start_date, end_date = scene['start_time'], scene['end_time']
    temporal = (start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'))
    results = earthaccess.search_data(short_name=short_name, temporal=temporal)
    files = sorted(earthaccess.download(results))

    return str(files[-1])


def get_epsg_code(reference_path: str) -> int:
    """Get the UTM EPSG code of the full reference scene.

    The zone is always taken from the whole frame, never a subset: the subset's geocode box is
    snapped onto the full-frame grid, which only works in the frame's projection.

    Args:
        reference_path: Path of the reference scene (full product or skeleton).

    Returns:
        epsg_code: UTM EPSG code of the full scene's centroid.
    """
    poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
    _, _, zone_number, zone_letter = utm.from_latlon(poly.centroid.y, poly.centroid.x)
    epsg_base = 32600 if zone_letter >= 'N' else 32700  # northern vs southern hemisphere
    return epsg_base + zone_number


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
    # Compatible by name = same track, direction, frame and mode. The polarization field (index 4)
    # is left out: check_polarizations compares the RSLCs' polarizations with the runconfig's.
    reference_fields, secondary_fields = reference['common'].split('_'), secondary['common'].split('_')
    del reference_fields[4], secondary_fields[4]
    if reference_fields != secondary_fields:
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


def check_polarizations(reference: dict[str, list[str]], secondary: dict[str, list[str]], template_yaml: Path) -> None:
    """Check that both RSLCs hold the polarizations the runconfig processes.

    Where the runconfig lists polarizations for a frequency, each must be in both RSLCs. Where it
    lists none, isce3 processes every polarization, so both RSLCs must hold the same ones.

    Args:
        reference: Reference RSLC polarizations per frequency group, from get_polarizations.
        secondary: Secondary RSLC polarizations per frequency group, from get_polarizations.
        template_yaml: Path of the GUNW runconfig.

    Raises:
        ValueError: When a processed polarization is missing from either RSLC.
    """
    processing = yaml.safe_load(Path(template_yaml).read_text())['runconfig']['groups']['processing']
    listed = (processing.get('input_subset') or {}).get('list_of_frequencies') or {}
    # With no frequencies listed, isce3 processes every frequency of the reference.
    for letter in listed or [frequency[-1] for frequency in reference]:
        reference_pols = reference.get(f'frequency{letter}', [])
        secondary_pols = secondary.get(f'frequency{letter}', [])
        wanted = listed.get(letter)
        if wanted:
            for pol in wanted:
                if pol not in reference_pols or pol not in secondary_pols:
                    raise ValueError(
                        f'The runconfig processes frequency{letter} {pol}, but the reference holds '
                        f'{reference_pols} and the secondary holds {secondary_pols}'
                    )
        elif sorted(reference_pols) != sorted(secondary_pols):
            raise ValueError(
                f'The runconfig processes every frequency{letter} polarization, but the reference holds '
                f'{reference_pols} and the secondary holds {secondary_pols}'
            )


def process_isce3(
    reference_scene: str, secondary_scene: str, subset: list[float] | None = None, fast: bool = False
) -> Path:
    """Run the isce3 InSAR workflow to produce a GUNW for a pair of RSLCs.

    Everything is staged and processed in a folder named for the product, so several runs
    in one working directory don't overwrite each other's files.

    Args:
        reference_scene: Name of one input scene (ordered internally against the other).
        secondary_scene: Name of the other input scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.
        fast: Run isce3 through :func:`hyp3_isce3.isce3_fast.fast_run`, which is quicker but whose
            products are not bit-identical to the standard run's.

    Returns:
        zip_path: Absolute path of the zip holding the GUNW h5 file and its runconfig.
    """
    # Use the earlier acquisition as the reference regardless of the order passed in.
    reference_scene, secondary_scene = earlier_granule_first(reference_scene, secondary_scene)
    product_id = get_product_id(reference_scene, secondary_scene)

    run_dir = Path(product_id).resolve()
    run_dir.mkdir(exist_ok=True)
    with contextlib.chdir(run_dir):
        zip_name = _run_pair(reference_scene, secondary_scene, subset, fast)
    return run_dir / zip_name


def _init_staging_worker() -> None:
    """Set up a staging worker process: the parent's log format and its own Earthdata session."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO
    )
    earthaccess.login()


def _run_pair(reference_scene: str, secondary_scene: str, subset: list[float] | None, fast: bool = False) -> str:
    """Stage the inputs, run isce3, and zip the GUNW, all in the current directory.

    Args:
        reference_scene: Name of the reference (earlier) scene.
        secondary_scene: Name of the secondary (later) scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.
        fast: Run isce3 with the speedups in :mod:`hyp3_isce3.isce3_fast` instead of the standard workflow.

    Returns:
        zip_name: File name of the zip holding the GUNW h5 file and its runconfig.
    """
    earthaccess.login()
    scenes = (reference_scene, secondary_scene)
    # The reference and secondary streams run side by side in worker processes. Threads would not
    # help: h5py holds one lock per process, so two scenes would still be read one at a time.
    with ProcessPoolExecutor(max_workers=8, initializer=_init_staging_worker) as pool:
        # These depend only on the scene names, so they download while the rest is staged.
        tropo_jobs = [pool.submit(get_tropo, scene, subset) for scene in scenes]
        orbit_jobs = [pool.submit(get_orbit, scene) for scene in scenes]
        tec_job = pool.submit(get_tec, reference_scene)
        # The JPL runconfig is both our config template (its tail) and the source of the
        # multilook looks the crop aligns to; download once and reuse for both.
        template_job = pool.submit(download_yaml, decode_rslc_name(reference_scene)['search_keyword'])

        if subset:
            # Stream a metadata-only skeleton instead of the full RSLC; it stands in for the
            # product in every pre-crop step, and the image window is streamed by crop_streamed.
            reference_path, secondary_path = (str(path) for path in pool.map(stream_skeleton, scenes))
        else:
            reference_path = download_rslc(reference_scene)
            secondary_path = download_rslc(secondary_scene)

        # When subsetting, stage the water mask, weather models, and DEM over the AOI only
        # (orbit/tec are temporal, so they are unaffected by the subset).
        watermask_job = pool.submit(mosaic_tiles, 'NISAR_WATERMASK', reference_path, subset, 'watermask.tif')

        epsg_code = get_epsg_code(reference_path)
        dem_path = mosaic_tiles('NISAR_DEM', reference_path, subset, 'dem.tif')
        template_yaml = template_job.result()
        # Stop here, before the crops, if either RSLC lacks a polarization the runconfig processes.
        check_polarizations(
            get_polarizations(SLC(hdf5file=reference_path)),
            get_polarizations(SLC(hdf5file=secondary_path)),
            template_yaml,
        )

        # Crop the RSLCs to the AOI before processing so the radar-domain steps run on a
        # small patch; the crop floors its origin to the multilook looks (see crop_rslc).
        subset_utm = None
        if subset:
            # Reproject the AOI to the output UTM box and snap it to the geocode grid so the subset's
            # pixels coincide with a full-frame run's (else a sub-pixel offset re-rolls speckle in
            # decorrelated areas).
            subset_utm = geocode_subset_box(subset, epsg_code, template_yaml)
            az_step, rg_step, rg_looks_b = get_multilook_looks(template_yaml)
            looks = {'az_looks': az_step, 'rg_looks': rg_step, 'rg_looks_b': rg_looks_b}
            # Stream each RSLC's AOI window into a cropped <scene>.h5, windowed independently from
            # its own orbit; only the overlapping chunks of the processed polarizations are pulled.
            polarizations = get_processed_polarizations(template_yaml)
            crop_jobs = [
                pool.submit(crop_streamed, scene, skeleton, subset, dem_path, polarizations=polarizations, **looks)
                for scene, skeleton in zip(scenes, (reference_path, secondary_path))
            ]
            reference_path, secondary_path = (job.result() for job in crop_jobs)

        reference_tropo, secondary_tropo = (job.result() for job in tropo_jobs)
        reference_orbit, secondary_orbit = (job.result() for job in orbit_jobs)
        tec_path = tec_job.result()
        watermask = watermask_job.result()

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

    if fast:
        # Same steps and runconfig with idle cores put to work; not always bit-identical to the standard run.
        isce3_fast.fast_run(insar_runcfg.cfg, out_paths, yaml_path)
    else:
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


def main() -> None:
    """Entrypoint for the GUNW workflow (++process gunw)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    parser.add_argument(
        'granules',
        type=str.split,
        nargs='+',
        help='NISAR RSLC granule names, or paths of local RSLC h5 files',
    )
    parser.add_argument(
        '--subset',
        type=nullable_subset_list,
        nargs='*',
        help='Optional WGS84 bounding box to subset the output GUNW (LON_MIN LAT_MIN LON_MAX LAT_MAX)',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run isce3 with parallel and overlapped steps. Quicker, but the GUNW is not bit-identical to a SDS run',
    )

    args = parser.parse_args()

    args.granules = [item for sublist in args.granules for item in sublist]
    # Make local RSLC paths absolute, as the run changes into its own working folder.
    args.granules = [str(Path(g).resolve()) if Path(g).is_file() else g for g in args.granules]
    if len(args.granules) != 2:
        parser.error('Must provide exactly two granules')

    args.subset = parse_subset(args.subset)

    # process_isce3 orders the pair itself (earlier acquisition becomes the reference).
    granule_1, granule_2 = args.granules
    product_file = process_isce3(granule_1, granule_2, subset=args.subset, fast=args.fast)

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
