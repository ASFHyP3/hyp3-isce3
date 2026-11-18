"""Focus a GCOV into an RSLC on a GPU with the isce3 focus workflow."""

import argparse
import logging
import shutil
from pathlib import Path

import asf_search as asf
import earthaccess
import h5py
import s3fs
import yaml
from hyp3lib.aws import upload_file_to_s3
from nisar.products.writers.GcovWriter import GcovWriter
from nisar.workflows import gcov as gcov_workflow
from nisar.workflows.gcov_runconfig import GCOVRunConfig

import hyp3_isce3
from hyp3_isce3.cli import nullable_subset_list, parse_subset
from hyp3_isce3.dem import mosaic_tiles
from hyp3_isce3.process import (
    crop_streamed,
    decode_rslc_name,
    download_rslc,
    geocode_subset_box,
    get_epsg_code,
    get_orbit,
    get_tec,
    stream_skeleton,
)
from hyp3_isce3.search import search_nisar


log = logging.getLogger(__name__)


def get_config(
    rslc_path: str,
    orbit_path: str | None,
    dem_path: str,
    tec_path: str,
    template_yaml: Path,
    frequencies: list[str],
    subset_utm: tuple[float, float, float, float] | None = None,
    output_epsg: int | None = None,
    polarizations: list[str] | None = None,
) -> Path:
    """Create a configuration file for isce3 from a production GCOV runconfig.

    The production runconfig (from :func:`download_yaml`) supplies every processing setting and
    the release metadata (CRID, product versions, DOIs). Only the fields that differ for our run
    are overwritten: input and ancillary paths, the ASF on-demand identity, the granule ID
    pattern for this pair, and (for subsets) the output grid.

    Args:
        rslc_path: Path of the RSLC scene.
        orbit_path: Path of the orbit.
        dem_path: Path of the DEM file.
        tec_path: Path of the TEC file for the scene.
        template_yaml: Path of the downloaded production runconfig (from :func:`download_yaml`).
        frequencies: List of frequencies to process - can be 'A' and/or 'B'.
        subset_utm: Optional (xmin, ymin, xmax, ymax) output box in UTM meters.
        output_epsg: EPSG code of ``subset_utm``; written into the geocode blocks so the corners and projection stay consistent.
            Without a subset it fills only a blank output_epsg (as in the fallback runconfig).
        polarizations: The desired polarizations to process.

    Returns:
        yaml_file: Path of the configuration file.
    """
    runconfig = yaml.safe_load(Path(template_yaml).read_text())
    groups = runconfig['runconfig']['groups']

    # Rebuild the input and ancillary groups from our files so none of production's job-local
    # paths can leak through; only the ancillary descriptions are carried over.
    groups['input_file_group'] = {
        'input_file_path': rslc_path,
    }
    descriptions = {
        key: value
        for key, value in groups.get('dynamic_ancillary_file_group', {}).items()
        if key.endswith('_description')
    }
    groups['dynamic_ancillary_file_group'] = {
        'dem_file': dem_path,
        'orbit_file': orbit_path,
        'tec_file': tec_path,
        **descriptions,
    }
    # Output and scratch locations are ours too, relative to the working directory.
    groups['product_path_group'] = {
        'product_path': 'output',
        'scratch_path': 'scratch',
        'sas_output_file': 'output/gcov.h5',
        'product_counter': 1,
        'qa_output_dir': 'qa_gcov',
    }

    input_subset = {freq: polarizations if polarizations else None for freq in frequencies}

    groups['processing']['input_subset']['list_of_frequencies'] = input_subset

    # Keep production's CRID, product versions, and DOIs; label the product as ASF on-demand.
    executable = groups['primary_executable']
    executable['product_type'] = 'GCOV'
    # isce3 records 'OD' (on-demand) as processingType 'Custom'; production's 'PR' would read 'Nominal'.
    executable['processing_type'] = 'OD'
    executable['processing_center'] = 'A'
    # isce3 fills the {...} tokens; the rest identifies this pair (reference cycle, track,
    # direction, frame, secondary cycle) and the release, as in a production GCOV name.
    rslc = decode_rslc_name(rslc_path)
    crid, accuracy, coverage = rslc['center'].split('_')[:3]
    # A subset covers only part of the frame, so mark it partial ('P') per the NISAR naming convention.
    if subset_utm:
        coverage = 'P'
    # Fall back to the RSLC's release if the template carries no CRID.
    executable['composite_release_id'] = executable.get('composite_release_id') or crid
    executable['partial_granule_id'] = (
        f'NISAR_L2_OD_GCOV_{rslc["cycle"]}_{rslc["track"]}_'
        f'{rslc["direction"]}_{rslc["frame"]}_{{MODE}}_{{PO}}_'
        '{StartDateTime}_{EndDateTime}_'
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
        geocode = groups['processing']['geocode']
        postings = geocode['output_posting'].values()
        geocode['x_snap'] = max(posting['x_posting'] for posting in postings)
        geocode['y_snap'] = max(posting['y_posting'] for posting in postings)

    yaml_file = Path('gcov.yaml')
    yaml_file.write_text(yaml.safe_dump(runconfig, sort_keys=False))
    return yaml_file


def download_yaml(keyword: str) -> Path:
    """Download the production GCOV runconfig for a frame, or copy the fallback when there is none.

    The subset focus calls this too, so it snaps to the same multilook looks this workflow will use.

    Args:
        keyword: track_direction_frame of the frame, e.g. '085_D_069'.

    Returns:
        tmp_yaml: Path of the yaml file.
    """
    try:
        results = search_nisar('L2', 'GCOV', granule_name=f'*{keyword}*')
    except ValueError:
        # No production GCOV on this frame yet; use the shipped runconfig instead. Copy it,
        # since the caller deletes temp.yaml once it is consumed.
        log.warning(f'No production GCOV config found for frame {keyword}; using the fallback runconfig.')
        fallback = Path(hyp3_isce3.__file__).parent / 'schemas' / 'gcov_fallback.yaml'
        return Path(shutil.copy(fallback, 'temp.yaml'))
    gcov = results[0].data_links()[0].split('/')[-2]
    res = asf.granule_search(gcov)
    yaml_url = res.find_urls(pattern=r'.yaml')[0]
    asf.download_url(url=yaml_url, path='./', filename='temp.yaml')

    tmp_yaml = Path('temp.yaml')

    return tmp_yaml


def process_gcov(
    scene: str,
    frequencies: list[str],
    polarizations: list[str] | None = None,
    subset: list[float] | None = None,
    s3_uri: str | None = None,
) -> str:
    """Create a GCOV product from an RSLC using ISCE3.

    Args:
        scene: Name of the reference scene or s3 URI for scene.
        polarizations: List of polarizations to process.
        frequencies: List of frequencies to process - can be 'A' and/or 'B'.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GCOV.
        s3_uri: Optional S3 URI to download input RSLC from.

    Returns:
        h5file: Path of the GCOV h5file.
    """
    earthaccess.login()

    try:
        orbit_path = get_orbit(scene)
    except RuntimeError:
        orbit_path = None

    download_path = s3_uri if s3_uri else scene

    if subset:
        # Stream a metadata-only skeleton instead of the full RSLC; it stands in for the
        # product in every pre-crop step, and the image window is streamed by crop_streamed.
        rslc_path = str(stream_skeleton(download_path))
    else:
        rslc_path = download_rslc(download_path)

    tec_path = get_tec(scene)
    epsg_code = get_epsg_code(rslc_path)
    dem_path = mosaic_tiles('NISAR_DEM', rslc_path, subset, 'dem.tif')

    template_yaml = download_yaml(decode_rslc_name(rslc_path)['search_keyword'])

    # Crop the RSLCs to the AOI before processing so the radar-domain steps run on a
    # small patch; the crop floors its origin to the multilook looks (see crop_rslc).
    subset_utm = None
    if subset:
        subset_utm = geocode_subset_box(subset, epsg_code, template_yaml)
        looks = {'az_looks': 1, 'rg_looks': 1, 'rg_looks_b': 1}
        # Stream each RSLC's AOI window into a cropped <scene>.h5, windowed
        # independently from its own orbit; only overlapping image chunks are pulled.
        rslc_path = crop_streamed(scene, rslc_path, subset, dem_path, **looks)

    yaml_path = get_config(
        rslc_path,
        orbit_path,
        dem_path,
        tec_path,
        template_yaml,
        frequencies,
        subset_utm,
        epsg_code,
        polarizations,
    )
    template_yaml.unlink()  # consumed by the looks read and the config build

    scratch_path = Path('scratch/')
    scratch_path.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(run_config_path=str(yaml_path), log_file=False)
    gcov_runconfig = GCOVRunConfig(args)

    _, orbit, timing_corrections = gcov_workflow.run(gcov_runconfig.cfg)

    with GcovWriter(runconfig=gcov_runconfig, orbit=orbit, timing_corrections_dict=timing_corrections) as gcov_obj:
        gcov_obj.populate_metadata()

    output = Path('output/gcov.h5')
    if not output.exists():
        raise RuntimeError('The GCOV file was not written!')

    with h5py.File(output, 'r') as gcov:
        band = next(iter(gcov['science']))
        granule_id = gcov[f'science/{band}/identification/granuleId'][()].decode()
    output = output.rename(f'{granule_id}.h5')
    yaml_path = yaml_path.rename(f'{granule_id}.rc.yaml')

    return f'{granule_id}.h5'


def determine_granule_uri_from_bucket(bucket: str, prefix: str) -> str:
    """Get the URI for a RSLC within a given bucket and bucket prefix.

    Args:
        bucket: AWS S3 bucket to search in.
        prefix: AWS s3 prefix within the bucket to search in.

    Returns: S3 URI of the granule
    """
    granule_folder = f's3://{bucket}/{prefix}'

    log.info(f'Looking for granule in {granule_folder}')

    s3_fs = s3fs.S3FileSystem(anon=False)
    matches = s3_fs.glob(f'{bucket}/{prefix.rstrip("/")}/*.h5')

    if not matches:
        raise FileNotFoundError(f'No H5 file found in s3://{bucket}/{prefix}')

    return f's3://{matches[0]}'


def main() -> None:
    """Entrypoint for the RSLC to GCOV workflow (++process gcov)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--granule', default='', help='NISAR RSLC granule name or S3 URI for the granule')
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument(
        '--frequency',
        type=str,
        help='Only process this frequency, "A" for full bandwidth and "B" for 5MHz band; default of None processes all.',
        default=None,
    )
    parser.add_argument(
        '--polarizations',
        nargs='+',
        help='Only process these polarizations, in every frequency (e.g. HH); default all',
    )
    parser.add_argument(
        '--subset',
        type=nullable_subset_list,
        nargs='*',
        help='Optional WGS84 bounding box to focus only (LON_MIN LAT_MIN LON_MAX LAT_MAX); streams just its pulses. '
        'On quad-pol L0Bs, also pass --polarizations with a single transmit (e.g. HH HV)',
    )

    args = parser.parse_args()
    args.subset = parse_subset(args.subset)

    if not args.granule:
        try:
            assert args.bucket and args.bucket_prefix
        except AssertionError:
            raise ValueError('Either `--granule` or `--bucket` and `--bucket-prefix` must be provided.')
        scene = determine_granule_uri_from_bucket(args.bucket, args.bucket_prefix)

    scene = args.granule
    s3_uri = None
    if scene.startswith('s3'):
        s3_uri = scene
        scene = s3_uri.split('/')[-1]

    freqs = args.granule.split('_')[8]
    if freqs == '0505' or freqs == '0005':
        log.info('Detected 5MHz product, only frequency B will be processed.')
        frequencies = ['B']
    elif args.frequency:
        frequencies = [args.frequency]
    else:
        frequencies = ['A', 'B']

    product_file = process_gcov(scene, frequencies, args.polarizations, args.subset, s3_uri=s3_uri)

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
