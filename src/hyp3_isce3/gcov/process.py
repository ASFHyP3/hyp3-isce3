"""Focus a NISAR L0B into an RSLC on a GPU with the isce3 focus workflow."""

import argparse
import logging
import shutil
from pathlib import Path

import asf_search as asf
import earthaccess
import yaml
from hyp3lib.aws import upload_file_to_s3
from nisar.workflows import gcov
from nisar.workflows.gcov_runconfig import GCOVRunConfig

import hyp3_isce3
from hyp3_isce3.cli import nullable_subset_list, parse_subset
from hyp3_isce3.dem import mosaic_tiles
from hyp3_isce3.process import (
    crop_streamed,
    decode_rslc_name,
    geocode_subset_box,
    get_epsg_code,
    get_multilook_looks,
    get_orbit,
    get_tec,
    stream_skeleton,
)
from hyp3_isce3.search import find_rslc, search_nisar


log = logging.getLogger(__name__)


def download_rslc(granule_name: str, bucket: str | None = None, bucket_prefix: str | None = None) -> str:
    """Download RSLC product, or use it as is when it is already a local file.

    Args:
        granule_name: Name of the scene, or the path of a local RSLC h5 file.
        bucket: An optional bucket containing the product to be downloaded.
        bucket_prefix: An optional bucket prefix where the product to be downloaded is located.

    Returns:
        h5file_path: Path of the h5 file.
    """
    # A local RSLC (e.g. one focused with ++process focus_l0b) is not in CMR.
    if Path(granule_name).is_file():
        return granule_name
    # Same lookup as the streaming path; keep only the .h5 in case the granule ships other files.
    if bucket and bucket_prefix:
        pass
    else:
        files = earthaccess.download([find_rslc(granule_name)])
        h5_files = [f for f in files if str(f).endswith('.h5')]
        if not h5_files:
            raise RuntimeError(f'No .h5 file downloaded for {granule_name}')
    return str(h5_files[0])


def get_config(
    rslc_path: str,
    orbit_path: str,
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
        rslc_path: Path of the reference scene.
        orbit_path: Path of the reference orbit.
        dem_path: Path of the DEM file.
        tec_path: Path of the TEC file for the reference scene.
        template_yaml: Path of the downloaded production runconfig (from :func:`download_yaml`).
        frequencies: List of frequencies to process - can be 'A' and/or 'B'.
        subset_utm: Optional (xmin, ymin, xmax, ymax) output box in UTM meters.
        output_epsg: EPSG code of ``subset_utm``; written into the geocode blocks so the corners and projection stay consistent.
            Without a subset it fills only a blank output_epsg (as in the fallback runconfig).
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GCOV.
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

    for freq in frequencies:
        if polarizations:
            groups['processing']['input_subset']['list_of_frequencies'][freq] = polarizations
        else:
            groups['processing']['input_subset']['list_of_frequencies'][freq] = ''

    # Keep production's CRID, product versions, and DOIs; label the product as ASF on-demand.
    executable = groups['primary_executable']
    # Write RIFG and RUNW alongside the GUNW as output/<PRODUCT>_product.h5 (with 'GUNW' alone,
    # isce3 puts them in scratch and the GUNW at sas_output_file), which process_isce3 expects.
    executable['product_type'] = 'GCOV'
    # isce3 records 'OD' (on-demand) as processingType 'Custom'; production's 'PR' would read 'Nominal'.
    executable['processing_type'] = 'OD'
    executable['processing_center'] = 'A'
    # isce3 fills the {...} tokens; the rest identifies this pair (reference cycle, track,
    # direction, frame, secondary cycle) and the release, as in a production GUNW name.
    rslc = decode_rslc_name(rslc_path)
    crid, accuracy, coverage = rslc['center'].split('_')[:3]
    # A subset covers only part of the frame, so mark it partial ('P') per the NISAR naming convention.
    if subset_utm:
        coverage = 'P'
    # Fall back to the reference RSLC's release if the template carries no CRID.
    executable['composite_release_id'] = executable.get('composite_release_id') or crid
    executable['partial_granule_id'] = (
        f'NISAR_{{Level}}_OD_{{ProductType}}_{rslc["cycle"]}_{rslc["track"]}_'
        f'{rslc["direction"]}_{rslc["frame"]}_{{MODE}}_{{PO}}_'
        '{RefStartDateTime}_{RefEndDateTime}_'
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

    yaml_file = Path('gcov.yaml')
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
        results = search_nisar('L2', 'GCOV', granule_name=f'*{keyword}*')
    except ValueError:
        # No production GUNW on this frame yet; use the shipped runconfig instead. Copy it,
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
) -> Path:
    """Get Polygon for reference scene.

    Args:
        scene: Name of the reference scene.
        polarizations: List of polarizations to process.
        frequencies: List of frequencies to process - can be 'A' and/or 'B'.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.

    Returns:
        h5file: Path of the GUNW h5file.
    """
    earthaccess.login()

    try:
        orbit_path = get_orbit(scene)
    except RuntimeError:
        orbit_path = None

    if subset:
        # Stream a metadata-only skeleton instead of the full RSLC; it stands in for the
        # product in every pre-crop step, and the image window is streamed by crop_streamed.
        rslc_path = str(stream_skeleton(scene))
    else:
        rslc_path = download_rslc(scene)

    tec_path = get_tec(scene)
    epsg_code = get_epsg_code(rslc_path)
    dem_path = mosaic_tiles('NISAR_DEM', rslc_path, subset, 'dem.tif')

    template_yaml = download_yaml(decode_rslc_name(rslc_path)['search_keyword'])

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

    gcov.run(cfg=gcov_runconfig.cfg)

    output = Path('output/gcov.h5')
    if not output.exists():
        raise RuntimeError('The RSLC file was not written!')

    return output


def main() -> None:
    """Entrypoint for the L0B focus workflow (++process focus_l0b)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('granule', help='NISAR L0B (RRSD) granule name')
    parser.add_argument(
        '--frequencies',
        nargs='+',
        help='Only focus these frequencies, "A" for full bandwidth and "B" for 5MHz band; default all.',
    )
    parser.add_argument(
        '--polarizations',
        nargs='+',
        help='Only focus these polarizations, in every frequency (e.g. HH for a GUNW-only run); default all',
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

    if args.granule.split('_')[8] == '0505':
        frequencies = ['B']
    else:
        frequencies = args.frequencies if args.frequencies else ['A', 'B']

    product_file = process_gcov(args.granule, frequencies, args.polarizations, args.subset)

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
