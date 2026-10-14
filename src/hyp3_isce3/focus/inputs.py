"""Find the L0B inputs and the runconfig template for focusing one frame."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import asf_search as asf
import earthaccess
import shapely
import yaml

import hyp3_isce3
from hyp3_isce3.focus.ancillary import get_frame_polygon
from hyp3_isce3.search import search_nisar


log = logging.getLogger(__name__)


def decode_l0b_name(name: str) -> dict:
    """Decode a NISAR L0B granule name (or file path) into its relevant fields.

    e.g. NISAR_L0_PR_RRSD_032_128_D_161S_20261009T021028_20261009T021116_P05023_F_J_001

    Args:
        name: L0B granule name, optionally with a directory/extension (e.g. a downloaded .h5).

    Returns:
        The fields the focus workflow uses, keyed by name.
    """
    parts = Path(name).stem.split('_')
    return {
        'cycle': parts[4],
        'track': parts[5],
        'direction': parts[6],  # ascending/descending flag
        'start_time': datetime.strptime(parts[8], '%Y%m%dT%H%M%S'),
        'end_time': datetime.strptime(parts[9], '%Y%m%dT%H%M%S'),
        'crid': parts[10],  # composite release ID
    }


def find_l0b(granule_name: str) -> earthaccess.DataGranule:
    """Find a NISAR L0B granule by name, whichever maturity collection holds it."""
    return search_nisar('L0B', 'RRSD', readable_granule_name=granule_name)[0]


def download_l0b(granule: earthaccess.DataGranule) -> str:
    """Download an L0B granule and return the path of its raw-data .h5 (not the QA stats file)."""
    name = granule['umm']['GranuleUR']
    files = earthaccess.download([granule])
    h5_files = [f for f in files if Path(f).name == f'{name}.h5']
    if not h5_files:
        raise RuntimeError(f'No .h5 file downloaded for {name}')
    return str(h5_files[0])


def l0b_footprint(granule: earthaccess.DataGranule) -> shapely.Polygon:
    """Read an L0B granule's ground footprint from its CMR metadata."""
    geometry = granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
    points = geometry['GPolygons'][0]['Boundary']['Points']
    return shapely.Polygon([(p['Longitude'], p['Latitude']) for p in points])


def find_production_runconfig(l0b_name: str, frame: int) -> Path | None:
    """Download the runconfig of the production RSLC focused from this L0B on this frame.

    Args:
        l0b_name: Name of the L0B granule.
        frame: Frame number.

    Returns:
        Path of the downloaded runconfig, or None if no production RSLC on the frame used this L0B.
    """
    l0b = decode_l0b_name(l0b_name)
    pattern = f'NISAR_L1_*_RSLC_{l0b["cycle"]}_{l0b["track"]}_{l0b["direction"]}_{frame:03d}_*'
    try:
        results = search_nisar('L1', 'RSLC', granule_name=pattern)
    except ValueError:
        return None

    for granule in results:
        rslc_name = granule['umm']['GranuleUR']
        yaml_url = asf.granule_search(rslc_name).find_urls(pattern=r'\.rc\.yaml')[0]
        template = Path('temp.yaml')
        # asf_search skips the download if the file exists, so clear any leftover from a crashed run.
        template.unlink(missing_ok=True)
        asf.download_url(url=yaml_url, path='./', filename=template.name)
        # An RSLC frame can span two L0Bs; keep this runconfig only if the requested L0B is among its inputs.
        groups = yaml.safe_load(template.read_text())['runconfig']['groups']
        inputs = [Path(p).stem for p in groups['input_file_group']['input_file_path']]
        if l0b_name in inputs:
            log.info(f'Using the production runconfig of {rslc_name}')
            return template
        template.unlink()
    return None


def get_template(l0b_name: str, frame: int) -> tuple[Path, bool]:
    """Get the runconfig template: production if one exists for this L0B and frame, else the fallback.

    The fallback focuses the whole L0B (its output_grid times are blank), so its frame-specific
    geometry and granule ID are filled in here from the L0B name and the track-frame database.

    Args:
        l0b_name: Name of the L0B granule.
        frame: Frame number.

    Returns:
        template: Path of the runconfig template (temp.yaml).
        is_production: Whether the template came from a production RSLC.
    """
    template = find_production_runconfig(l0b_name, frame)
    if template is not None:
        return template, True

    log.warning(f'No production RSLC found from {l0b_name} on frame {frame}; focusing the whole L0B.')
    fallback = Path(hyp3_isce3.__file__).parent / 'schemas' / 'rslc_fallback.yaml'
    template = Path(shutil.copy(fallback, 'temp.yaml'))

    l0b = decode_l0b_name(l0b_name)
    runconfig = yaml.safe_load(template.read_text())
    groups = runconfig['runconfig']['groups']
    groups['geometry'].update(
        {
            'relative_orbit_number': int(l0b['track']),
            'frame_number': frame,
            'cycle_number': int(l0b['cycle']),
            'orbit_direction': 'Ascending' if l0b['direction'] == 'A' else 'Descending',
            'track_frame_polygon': get_frame_polygon(int(l0b['track']), frame),
        }
    )
    # Same layout as a production RSLC ID; isce3 fills the {...} tokens.
    groups['primary_executable']['composite_release_id'] = l0b['crid']
    groups['primary_executable']['partial_granule_id'] = (
        f'NISAR_L1_OD_RSLC_{l0b["cycle"]}_{l0b["track"]}_{l0b["direction"]}_{frame:03d}_{{MODE}}_{{POLE}}_A_'
        f'{{StartDateTime}}_{{EndDateTime}}_{l0b["crid"]}_N_{{C}}_A_001'
    )
    # The L0B input stays in the template so the caller reads it the same way as a production one.
    groups['input_file_group']['input_file_path'] = [f'{l0b_name}.h5']
    template.write_text(yaml.safe_dump(runconfig, sort_keys=False))
    return template, False
