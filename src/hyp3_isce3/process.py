"""isce3 processing."""

import argparse
import logging
import shutil
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import earthaccess
import utm
import yaml
from hyp3lib.dem import prepare_dem_geotiff
from nisar.workflows import h5_prep, insar, stage_dem
from nisar.workflows.insar_runconfig import InsarRunConfig
from osgeo import gdal, ogr, osr

import hyp3_isce3
from hyp3_isce3.crop_rslc import crop_streamed, geocode_subset_box, stream_skeleton


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


def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` in place (dict values merged, not replaced)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _expand_dotted(overrides: dict) -> dict:
    """Expand dotted keys ({'a.b': v}) into nested dicts ({'a': {'b': v}}), recursively.

    Branches that collide (a dotted and a nested key sharing a prefix) are merged, not
    overwritten, so ``{'processing.crossmul.x': 1, 'processing': {'y': 2}}`` keeps both.
    """
    expanded: dict = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            value = _expand_dotted(value)
        nested = value
        for part in reversed(key.split('.')):
            nested = {part: nested}
        _deep_update(expanded, nested)
    return expanded


def _merge_existing(config: dict, overrides: dict, prefix: str = '') -> None:
    """Deep-merge ``overrides`` into ``config`` in place; every key must already exist.

    A path that does not resolve to an existing key raises KeyError -- that validates the
    request and blocks injecting keys the workflow would silently ignore.
    """
    for key, value in overrides.items():
        path = f'{prefix}{key}'
        if key not in config:
            raise KeyError(f'override key does not exist in the runconfig: {path}')
        if isinstance(value, dict) and isinstance(config[key], dict):
            _merge_existing(config[key], value, prefix=f'{path}.')
        elif isinstance(value, dict):
            raise KeyError(f'override targets a leaf, not a section: {path}')
        else:
            config[key] = value


def apply_overrides(template_yaml: Path, overrides: dict) -> None:
    """Deep-merge ``overrides`` into the runconfig at ``template_yaml``, in place.

    Overrides are relative to ``runconfig.groups`` (e.g. ``processing.crossmul.range_looks``)
    and may be written nested or with dotted keys. Only keys that already exist in the
    downloaded runconfig may be set; an unknown path raises so typos and junk keys fail fast.

    Args:
        template_yaml: Path of the downloaded runconfig to modify.
        overrides: Mapping of runconfig keys (under ``groups``) to new values.
    """
    config = yaml.safe_load(Path(template_yaml).read_text())
    _merge_existing(config['runconfig']['groups'], _expand_dotted(overrides))
    # Dump with 4-space indent to match the JPL runconfig / schema header indentation, so
    # get_config's line-based splice keeps groups children (product_path_group, processing, ...)
    # nested under `groups`. yaml's default 2-space indent would drop them a level and break the
    # spliced runconfig's structure (isce3 schema validation then fails).
    Path(template_yaml).write_text(yaml.safe_dump(config, sort_keys=False, indent=4))


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
    if len(results) == 0:
        # The reference scene's own cycle may never have been a GUNW reference (e.g. a custom
        # pair the operational catalog didn't process). The runconfig template is generic to the
        # track/frame, so fall back to any GUNW over the same track/dir/frame.
        frame_key = '_'.join(reference_path.split('_')[5:8])
        results = earthaccess.search_data(short_name=short_name, granule_name=f'*_{frame_key}_*')
    if len(results) == 0:
        raise ValueError(f'No {short_name} runconfig template found for {reference_path}')
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
    bbox = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
    results = earthaccess.search_data(short_name=short_name, bounding_box=bbox)
    files = sorted(earthaccess.download(results))

    # The download is a VRT mosaic that references sibling tile .tifs by path. Materialize it into
    # a single self-contained GeoTIFF clipped to the AOI (the same way get_dem does for the DEM),
    # so it has no external references and can be relocated/cached like every other ancillary.
    mask_path = 'watermask.tif'
    warped = gdal.Warp(mask_path, str(files[-1]), outputBounds=bbox, dstSRS='EPSG:4326')
    if warped is None:
        raise RuntimeError('Failed to warp the NISAR water mask into a standalone GeoTIFF')
    warped = None  # flush to disk

    return mask_path


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
        buf = 0.1  # degrees (~11 km)
        bbox = [subset[0] - buf, subset[1] - buf, subset[2] + buf, subset[3] + buf]
        poly, _ = stage_dem.determine_polygon(reference_path, bbox=bbox, bbox_epsg='4326')
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


def _staged(cache_dir: str | None, name: str, produce: Callable[[], str]) -> str:
    """Return ``cache_dir/name``, running ``produce`` only on a cache miss.

    Lets the demo reuse already-downloaded/cropped inputs across runs: the first run
    downloads or crops and stashes the result under ``name``; a later run with the same
    ``cache_dir`` finds it and skips the work. With ``cache_dir=None`` this is a no-op
    passthrough that just calls ``produce`` (the normal fetch-every-time behavior).

    Args:
        cache_dir: Directory of cached inputs, or None to disable caching.
        name: Deterministic filename for this artifact inside ``cache_dir``.
        produce: Zero-arg callback that downloads/crops and returns the produced path.

    Returns:
        Path (as str) of the cached (or freshly produced) file.
    """
    if cache_dir is None:
        return produce()
    target = Path(cache_dir) / name
    if target.exists():
        log.info('reusing cached %s', target)
        return str(target)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    produced = Path(produce())
    if produced.resolve() != target.resolve():
        shutil.move(str(produced), str(target))
    return str(target)


def process_isce3(
    reference_scene: str,
    secondary_scene: str,
    subset: list[float] | None = None,
    overrides: dict | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Get Polygon for reference scene.

    Args:
        reference_scene: Name of the reference scene.
        secondary_scene: Name of the secondary scene.
        subset: Optional WGS84 bounding box [lon_min, lat_min, lon_max, lat_max] to subset the output GUNW.
        overrides: Optional runconfig overrides (under ``groups``); only existing keys may be set.
        cache_dir: Optional directory of pre-fetched inputs. When set, every downloaded/cropped
            input (skeletons, cropped RSLCs, orbits, TEC, water mask, DEM) is stashed here on the
            first run and reused on later runs -- so a demo populated once reruns without
            re-downloading or re-cropping. The cropped RSLCs are stored as ``<scene>_sub.h5``,
            matching the demo subset script, so pre-cropped RSLCs are picked up directly.

    Returns:
        h5file: Path of the GUNW h5file.
    """
    product_id = get_product_id(reference_scene, secondary_scene)

    earthaccess.login()
    if subset:
        # Stream a metadata-only skeleton instead of the full RSLC; it stands in for the
        # product in every pre-crop step, and the image window is streamed by crop_streamed.
        reference_path = _staged(cache_dir, f'{reference_scene}_skel.h5', lambda: str(stream_skeleton(reference_scene)))
        secondary_path = _staged(cache_dir, f'{secondary_scene}_skel.h5', lambda: str(stream_skeleton(secondary_scene)))
    else:
        reference_path = _staged(cache_dir, f'{reference_scene}.h5', lambda: download_rslc(reference_scene))
        secondary_path = _staged(cache_dir, f'{secondary_scene}.h5', lambda: download_rslc(secondary_scene))

    # When subsetting, stage the water mask and DEM over the AOI only (orbit/tropo/tec
    # are temporal, so they are unaffected by the subset). get_watermask materializes a
    # standalone GeoTIFF, so it caches like the DEM.
    watermask = _staged(cache_dir, 'watermask.tif', lambda: get_watermask(reference_path, subset))

    reference_orbit = _staged(cache_dir, f'{reference_scene}_orbit.xml', lambda: get_orbit(reference_scene))
    secondary_orbit = _staged(cache_dir, f'{secondary_scene}_orbit.xml', lambda: get_orbit(secondary_scene))

    # Demo config: tropo + iono corrections are disabled so runs stay fast and focus on the
    # interferogram/parameter behavior. Skipping tropo also avoids the ~2 GB-each ECMWF
    # downloads -- get_config still needs a path for the tropo fields, so reuse the
    # already-fetched watermask (an existing file that validates); troposphere_delay.enabled is
    # forced off below so it is never read. TEC is always fetched: it is tiny (~20 MB) and isce3
    # validates it as JSON at config load even when the ionosphere step is disabled.
    run_ionosphere = False
    run_troposphere = False
    reference_tropo = get_tropo(reference_scene) if run_troposphere else watermask
    secondary_tropo = get_tropo(secondary_scene) if run_troposphere else watermask

    # Keyed on the reference scene: TEC is time-specific, and isce3 rejects a TEC file that does not
    # bracket the radar sensing window. An unkeyed name would serve one run's TEC to a later run with
    # a different reference date, which fails the temporal-coverage check at config load.
    tec_path = _staged(cache_dir, f'{reference_scene}_tec.json', lambda: get_tec(reference_scene))

    scene_polygon, epsg_code = get_scene_polygon(reference_path, subset)
    dem_path = _staged(cache_dir, 'dem.tif', lambda: get_dem(scene_polygon, epsg_code))
    # The runconfig schema references the DEM by the relative name ``dem.tif`` (it is the one
    # ancillary not threaded through get_config). When served from a cache_dir the DEM lives
    # elsewhere, so link it into the working dir to keep that relative reference resolvable.
    if Path('dem.tif').resolve() != Path(dem_path).resolve():
        dem_link = Path('dem.tif')
        if dem_link.is_symlink() or dem_link.exists():
            dem_link.unlink()
        dem_link.symlink_to(Path(dem_path).resolve())

    # The JPL runconfig is both our config template (its tail) and the source of the
    # crossmul looks the crop aligns to; download once and reuse for both. Keyed off the
    # granule name (not the file path) so a cache_dir prefix cannot shift its keyword split.
    #
    # Cached alongside the other ancillaries so a warm cache_dir needs no runconfig download.
    # Copied to a working file first: apply_overrides rewrites the template in place, and the
    # cached copy must stay pristine -- otherwise a later run with different overrides would
    # silently inherit this run's values for any key it does not set itself.
    cached_yaml = _staged(
        cache_dir, f'{reference_scene}_runconfig.yaml', lambda: str(download_yaml(reference_scene))
    )
    template_yaml = Path('runconfig_template.yaml')
    shutil.copy(cached_yaml, template_yaml)

    # Apply user runconfig overrides before anything reads the template, so the crossmul
    # looks and geocode grid the crop snaps to track any overridden values too.
    # Force the tropo/iono correction flags off when their steps are skipped, so isce3 does not
    # validate/read the placeholder (watermask) tropo/tec paths substituted above.
    overrides = dict(overrides or {})
    if not run_ionosphere:
        overrides.setdefault('processing.ionosphere_phase_correction.enabled', False)
    if not run_troposphere:
        overrides.setdefault('processing.troposphere_delay.enabled', False)
    if overrides:
        apply_overrides(template_yaml, overrides)

    # The optional coregistration-refinement steps run only when their runconfig enabled flag is
    # set, so a user can drop the (expensive) dense-offset/rubbersheet chain via --override and it
    # is actually skipped -- not just disabled in config while the step still executes.
    _proc = yaml.safe_load(Path(template_yaml).read_text())['runconfig']['groups']['processing']
    run_dense_offsets = bool(_proc['dense_offsets']['enabled'])
    run_offsets_product = bool(_proc['offsets_product']['enabled'])
    run_rubbersheet = bool(_proc['rubbersheet']['enabled'])
    run_fine_resample = bool(_proc['fine_resample']['enabled'])

    # Crop the RSLCs to the AOI before processing so the radar-domain steps run on a
    # small patch; the crop floors its origin to the crossmul looks (see crop_rslc).
    subset_utm = None
    if subset:
        # Reproject the AOI to the output UTM box and snap it to the geocode grid so the subset's
        # pixels coincide with a full-frame run's (else a sub-pixel offset re-rolls speckle in
        # decorrelated areas).
        subset_utm = geocode_subset_box(subset, epsg_code, template_yaml)
        az_looks, rg_looks = get_crossmul_looks(template_yaml)
        # Stream each RSLC's AOI window into a cropped <scene>_sub.h5, windowed
        # independently from its own orbit; only overlapping image chunks are pulled.
        # Cache the cropped products so demo reruns skip the (slow) streaming crop.
        reference_skel, secondary_skel = reference_path, secondary_path
        reference_path = _staged(
            cache_dir,
            f'{reference_scene}_sub.h5',
            lambda: crop_streamed(reference_scene, reference_skel, subset, dem_path, az_looks=az_looks, rg_looks=rg_looks),
        )
        secondary_path = _staged(
            cache_dir,
            f'{secondary_scene}_sub.h5',
            lambda: crop_streamed(secondary_scene, secondary_skel, subset, dem_path, az_looks=az_looks, rg_looks=rg_looks),
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
        'dense_offsets': run_dense_offsets,  # follows runconfig enabled flag (overridable)
        'offsets_product': run_offsets_product,
        'rubbersheet': run_rubbersheet,
        'fine_resample': run_fine_resample,
        'crossmul': True,
        'filter_interferogram': True,
        'unwrap': True,
        'ionosphere': run_ionosphere,  # demo: off
        'geocode': True,
        'solid_earth_tides': True,
        'baseline': True,
        'troposphere': run_troposphere,  # demo: off; gates the tropo download above too
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
