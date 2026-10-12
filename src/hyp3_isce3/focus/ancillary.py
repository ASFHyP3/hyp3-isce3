"""Download the ancillary files the RSLC focus workflow needs."""

import logging
from datetime import datetime
from pathlib import Path

import earthaccess
import geopandas as gpd
import shapely


log = logging.getLogger(__name__)

# CMR collection holding each ancillary runconfig field, for files the runconfig names.
ANCILLARY_COLLECTIONS = {
    'orbit': 'NISAR_OE',
    'pointing': 'NISAR_RP',
    'external_calibration': 'NISAR_ANC_AUX',
    'internal_calibration': 'NISAR_ANC_AUX',
    'antenna_pattern': 'NISAR_ANC_AUX',
}


def download_by_name(short_name: str, filename: str) -> str:
    """Download one ancillary file by its file name (the granule is named by the file's stem)."""
    results = earthaccess.search_data(short_name=short_name, granule_name=Path(filename).stem)
    files = [f for f in earthaccess.download(results) if Path(f).name == Path(filename).name]
    if not files:
        raise RuntimeError(f'{filename} not found in {short_name}')
    return str(files[0])


def download_newest_version(short_name: str, prefix: str) -> str:
    """Download the newest granule whose name starts with prefix; names end in a date and version, so sort works."""
    results = earthaccess.search_data(short_name=short_name, granule_name=f'{prefix}*')
    if not results:
        raise RuntimeError(f'No {prefix}* granule found in {short_name}')
    newest = max(results, key=lambda g: g['umm']['GranuleUR'])
    log.info(f'Using {newest["umm"]["GranuleUR"]}')
    return str(earthaccess.download([newest])[0])


def download_by_time(short_name: str, patterns: list[str], start: datetime, end: datetime) -> str:
    """Download the newest file covering [start, end], trying patterns from most to least accurate."""
    temporal = (start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'))
    for pattern in patterns:
        results = earthaccess.search_data(short_name=short_name, granule_name=pattern, temporal=temporal)
        if results:
            break
    else:
        raise RuntimeError(f'No {short_name} file matching {patterns} covers {temporal}')
    files = sorted(earthaccess.download(results))
    log.info(f'Using {Path(files[-1]).name}')
    return str(files[-1])


def download_named_ancillaries(ancillary_group: dict) -> dict:
    """Download the ancillary files a production runconfig names, so the focus matches production.

    Args:
        ancillary_group: The runconfig's dynamic_ancillary_file_group.

    Returns:
        Local path of each ancillary file, keyed by runconfig field; fields the runconfig leaves blank stay None.
    """
    paths = {}
    for field, short_name in ANCILLARY_COLLECTIONS.items():
        filename = ancillary_group.get(field)
        paths[field] = download_by_name(short_name, Path(filename).name) if filename else None
    return paths


def download_latest_ancillaries(start: datetime, end: datetime) -> dict:
    """Pick ancillary files for an L0B with no production runconfig: by time, and newest calibration.

    Args:
        start: L0B start time.
        end: L0B end time.

    Returns:
        Local path of each ancillary file, keyed by runconfig field.
    """
    return {
        # Most to least accurate: precise, medium, near-real-time, then forecast orbit ephemeris.
        'orbit': download_by_time('NISAR_OE', ['*POE*', '*MOE*', '*NOE*', '*FOE*'], start, end),
        # L-band radar pointing, most to least accurate: precise, near-real-time, then forecast.
        'pointing': download_by_time(
            'NISAR_RP', ['NISAR_ANC_L_PR_PRP*', 'NISAR_ANC_L_PR_NRP*', 'NISAR_ANC_L_PR_FRP*'], start, end
        ),
        'external_calibration': download_newest_version('NISAR_ANC_AUX', 'NISAR_LSAR_EXT_CAL_'),
        'internal_calibration': download_newest_version('NISAR_ANC_AUX', 'NISAR_LSAR_INT_CAL_'),
        'antenna_pattern': download_newest_version('NISAR_ANC_AUX', 'NISAR_ANTPAT_'),
    }


def get_frame_polygon(track: int, frame: int) -> str:
    """Read a frame's polygon from the newest NISAR track-frame database, as GeoJSON."""
    gpkg = download_newest_version('NISAR_ANC_AUX', 'NISAR_TrackFrame_L_')
    frames = gpd.read_file(gpkg)
    match = frames[(frames.track == track) & (frames.frame == frame)]
    if match.empty:
        raise ValueError(f'Track {track} frame {frame} not in {Path(gpkg).name}')
    return shapely.to_geojson(match.geometry.iloc[0])
