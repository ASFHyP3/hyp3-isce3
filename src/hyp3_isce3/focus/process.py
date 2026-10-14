"""Focus a NISAR L0B into an RSLC on a GPU with the isce3 focus workflow."""

import argparse
import contextlib
import logging
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import earthaccess
import h5py
import isce3
import numpy as np
import shapely
import yaml
from hyp3lib.aws import upload_file_to_s3
from nisar.mixed_mode import PolChannelSet
from nisar.products.readers.Raw import open_rrsd
from nisar.products.readers.Raw.Raw import RawBase
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows import focus as nisar_focus

from hyp3_isce3.dem import mosaic_tiles
from hyp3_isce3.focus.ancillary import download_latest_ancillaries, download_named_ancillaries
from hyp3_isce3.focus.inputs import decode_l0b_name, download_l0b, find_l0b, get_template, l0b_footprint


log = logging.getLogger(__name__)


def get_config(template_yaml: Path, l0b_paths: list[str], ancillaries: dict, dem_path: str) -> Path:
    """Create the focus runconfig from a production (or fallback) RSLC runconfig.

    The template supplies every processing setting, the frame geometry and output times, and the
    release metadata. Only the fields that differ for our run are overwritten: input and ancillary
    paths, output locations, the ASF on-demand identity, and the GPU switch.

    Args:
        template_yaml: Path of the runconfig template (from :func:`get_template`).
        l0b_paths: Local paths of the L0B files, in the template's order.
        ancillaries: Local path of each ancillary file, keyed by runconfig field.
        dem_path: Path of the DEM file.

    Returns:
        yaml_file: Path of the configuration file.
    """
    runconfig = yaml.safe_load(Path(template_yaml).read_text())
    groups = runconfig['runconfig']['groups']

    # Rebuild the input and ancillary groups from our files so none of production's job-local
    # paths can leak through; only the ancillary descriptions are carried over.
    groups['input_file_group'] = {'input_file_path': l0b_paths, 'qa_input_file': 'output/rslc.h5'}
    descriptions = {
        key: value
        for key, value in groups.get('dynamic_ancillary_file_group', {}).items()
        if key.endswith('_description')
    }
    groups['dynamic_ancillary_file_group'] = {'dem_file': dem_path, **ancillaries, **descriptions}
    groups['product_path_group'] = {
        'product_path': 'output',
        'scratch_path': 'scratch',
        'sas_output_file': 'output/rslc.h5',
        'sas_config_file': None,
        'product_counter': 1,
        'qa_output_dir': 'output',
    }

    # Keep production's CRID, product version, and DOI; label the product as ASF on-demand.
    executable = groups['primary_executable']
    executable['processing_type'] = 'OD'
    executable['processing_center'] = 'A'
    # Relabel the granule ID's processing type (field 2, 'PR') and producer (second-to-last field, 'J').
    fields = executable['partial_granule_id'].split('_')
    fields[2], fields[-2] = 'OD', 'A'
    executable['partial_granule_id'] = '_'.join(fields)

    # isce3 raises if no CUDA device is found, so a job landing on a CPU node fails immediately.
    groups['worker'] = {'gpu_enabled': True, 'gpu_id': 0}

    yaml_file = Path('focus.yaml')
    yaml_file.write_text(yaml.safe_dump(runconfig, sort_keys=False))
    return yaml_file


def get_frame_times(l0b_path: str, orbit_path: str, track_frame_polygon: str) -> tuple[datetime, datetime]:
    """Find the zero-Doppler time span of a frame polygon, to cut the focus output to that frame.

    Each polygon vertex (on the ellipsoid) is projected to radar time with geo2rdr; the frame spans
    the earliest to the latest vertex time. On production frames this lands within 1 s of the
    production output_grid times, which are whole seconds. A frame can extend past the L0B (production
    then focuses it from two L0Bs); the span is cut to the L0B's pulses, so the RSLC is partial. The span
    is then rounded inward to whole seconds, which (like production's times) lie on isce3's default
    azimuth lattice, so the output grid coincides with the production RSLC's.

    Args:
        l0b_path: Path of the L0B file.
        orbit_path: Path of the orbit XML file.
        track_frame_polygon: Frame polygon as GeoJSON.

    Returns:
        start_time: Zero-Doppler time of the frame's first line (UTC).
        end_time: Zero-Doppler time of the frame's last line (UTC).
    """
    raw = open_rrsd(l0b_path)
    wavelength = isce3.core.speed_of_light / raw.getCenterFrequency(raw.frequencies[0])
    side = raw.identification.lookDirection
    # Crop the orbit to the L0B's span, so geo2rdr's search starts on this pass of a multi-pass orbit
    # file. The margin keeps vertices of a frame that extends past the L0B inside the orbit.
    l0b = decode_l0b_name(Path(l0b_path).stem)
    margin = timedelta(seconds=60)
    orbit = load_orbit_from_xml(orbit_path)
    orbit = orbit.crop(
        isce3.core.DateTime(l0b['start_time'] - margin), isce3.core.DateTime(l0b['end_time'] + margin), 3
    )

    times = []
    for lon, lat in shapely.get_coordinates(shapely.from_geojson(track_frame_polygon)):
        llh = np.array([np.radians(lon), np.radians(lat), 0.0])
        t, _ = isce3.geometry.geo2rdr(llh, isce3.core.Ellipsoid(), orbit, isce3.core.LUT2d(), wavelength, side)
        times.append(t)
    start, end = min(times), max(times)

    # Cut the span to the pulses this L0B holds; the frame may continue into the neighboring L0B.
    _, pulse_times = raw.getPulseTimes(epoch=orbit.reference_epoch)
    if start < pulse_times[0] or end > pulse_times[-1]:
        log.warning('The frame extends past this L0B; the RSLC covers only the part of the frame it holds.')
        start, end = max(start, pulse_times[0]), min(end, pulse_times[-1])

    # Orbit times are seconds since its reference epoch; isce3 prints nanoseconds, Python keeps microseconds.
    epoch = datetime.fromisoformat(orbit.reference_epoch.isoformat()[:26]).replace(tzinfo=UTC)
    start_time, end_time = epoch + timedelta(seconds=start), epoch + timedelta(seconds=end)
    # Round inward to whole seconds: an explicit start_time skips isce3's azimuth-grid snap, and whole
    # seconds sit on that grid (1/PRF from midnight), so the lines line up with the production RSLC's.
    if start_time.microsecond:
        start_time = start_time.replace(microsecond=0) + timedelta(seconds=1)
    end_time = end_time.replace(microsecond=0)
    log.info(f'Frame zero-Doppler span from the frame polygon: {start_time} to {end_time}')
    return start_time, end_time


@contextlib.contextmanager
def only_polarizations(polarizations: list[str] | None) -> Iterator[None]:
    """Restrict the isce3 focus to the given polarizations, in every frequency the L0B carries.

    The focus runconfig has no channel selection: isce3 focuses every channel it finds through
    PolChannelSet.from_raw. Filtering there, rather than the output mode, also keeps the product's
    mixed-mode flag false, since isce3 sets that flag by comparing from_raw with the output mode.

    Args:
        polarizations: Polarizations to keep, e.g. ['HH']. None keeps every channel.
    """
    if not polarizations:
        yield
        return

    from_raw = PolChannelSet.__dict__['from_raw']

    def from_raw_subset(cls: type, raw: RawBase) -> PolChannelSet:
        channels = from_raw.__func__(cls, raw)
        kept = cls(channel for channel in channels if channel.pol in polarizations)
        if not kept:
            raise ValueError(f'None of {polarizations} are in the L0B, which has {sorted(c.pol for c in channels)}')
        return kept

    PolChannelSet.from_raw = classmethod(from_raw_subset)
    try:
        yield
    finally:
        PolChannelSet.from_raw = from_raw


def process_focus(l0b_name: str, frame: int, polarizations: list[str] | None = None) -> Path:
    """Focus one frame of an L0B into an RSLC.

    Everything is staged and processed in a folder named for the L0B and frame, so several runs
    in one working directory don't overwrite each other's files.

    Args:
        l0b_name: Name of the L0B granule.
        frame: Frame number to focus.
        polarizations: Only focus these polarizations (e.g. ['HH']); None focuses every channel.

    Returns:
        zip_path: Absolute path of the zip holding the RSLC h5 file and its runconfig.
    """
    run_dir = Path(f'{l0b_name}_F{frame:03d}').resolve()
    run_dir.mkdir(exist_ok=True)
    with contextlib.chdir(run_dir):
        zip_name = _run_focus(l0b_name, frame, polarizations)
    return run_dir / zip_name


def _run_focus(l0b_name: str, frame: int, polarizations: list[str] | None = None) -> str:
    """Stage the inputs, run the isce3 focus, and zip the RSLC, all in the current directory.

    Args:
        l0b_name: Name of the L0B granule.
        frame: Frame number to focus.
        polarizations: Only focus these polarizations (e.g. ['HH']); None focuses every channel.

    Returns:
        zip_name: File name of the zip holding the RSLC h5 file and its runconfig.
    """
    earthaccess.login()
    template_yaml, is_production = get_template(l0b_name, frame)
    groups = yaml.safe_load(template_yaml.read_text())['runconfig']['groups']

    # A production frame can span two L0Bs; fetch every one its runconfig lists.
    l0b_names = [Path(p).stem for p in groups['input_file_group']['input_file_path']]
    l0b_granules = [find_l0b(name) for name in l0b_names]
    l0b_paths = [download_l0b(granule) for granule in l0b_granules]
    log.info(f'Downloaded L0B: {[Path(p).name for p in l0b_paths]}')

    if is_production:
        ancillaries = download_named_ancillaries(groups['dynamic_ancillary_file_group'])
    else:
        l0b = decode_l0b_name(l0b_name)
        ancillaries = download_latest_ancillaries(l0b['start_time'], l0b['end_time'])
        # The fallback's output times are blank (the whole L0B); cut the output to the frame instead.
        start_time, end_time = get_frame_times(
            l0b_paths[0], ancillaries['orbit'], groups['geometry']['track_frame_polygon']
        )
        runconfig = yaml.safe_load(template_yaml.read_text())
        runconfig['runconfig']['groups']['processing']['output_grid'].update(
            {'start_time': start_time, 'end_time': end_time}
        )
        template_yaml.write_text(yaml.safe_dump(runconfig, sort_keys=False))

    # The DEM must cover the focused swath: the L0B footprints, trimmed to the frame.
    footprint = shapely.union_all([l0b_footprint(granule) for granule in l0b_granules])
    footprint = footprint.intersection(shapely.from_geojson(groups['geometry']['track_frame_polygon']))
    # mosaic_tiles reads only the bounding box when one is given (the path argument is not used then).
    dem_path = mosaic_tiles('NISAR_DEM', l0b_paths[0], list(footprint.bounds), 'dem.tif')

    yaml_path = get_config(template_yaml, l0b_paths, ancillaries, dem_path)
    template_yaml.unlink()  # consumed by the input lookups and the config build

    log.info(f'Running the isce3 focus workflow on the GPU, polarizations: {polarizations or "all"}')
    with only_polarizations(polarizations):
        nisar_focus.focus(nisar_focus.load_config(str(yaml_path)), str(yaml_path))

    output = Path('output/rslc.h5')
    if not output.exists():
        raise RuntimeError('The RSLC file was not written!')
    # Name the product by the granule ID isce3 wrote into it (mode, polarization, and times filled in).
    with h5py.File(output, 'r') as rslc:
        band = next(iter(rslc['science']))
        granule_id = rslc[f'science/{band}/identification/granuleId'][()].decode()
    output = output.rename(f'{granule_id}.h5')
    yaml_path = yaml_path.rename(f'{granule_id}.rc.yaml')

    with zipfile.ZipFile(f'{granule_id}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.write(str(output))
        zip_ref.write(str(yaml_path))
    return f'{granule_id}.zip'


def main() -> None:
    """Entrypoint for the L0B focus workflow (++process focus_l0b)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('granule', help='NISAR L0B (RRSD) granule name')
    parser.add_argument('--frame', type=int, required=True, help='NISAR frame number to focus')
    parser.add_argument(
        '--polarizations',
        nargs='+',
        help='Only focus these polarizations, in every frequency (e.g. HH for a GUNW-only run); default all',
    )
    args = parser.parse_args()

    product_file = process_focus(args.granule, args.frame, args.polarizations)

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
