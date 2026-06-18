"""Crop a NISAR L1 RSLC to a small radar-coordinate window.

The isce3 ``nisar.workflows.insar`` pipeline runs every pre-geocoding step in
radar geometry over the full swath, so a geocode bounding box does not make it
cheaper -- the inputs themselves must be small. This module maps an area of
interest (AOI) bounding box into an RSLC's radar grid and writes a cropped copy
holding only that (azimuth line, range sample) window.

The window is found by solving the four AOI corners into the radar grid with
isce3 ``geo2rdr`` (zero-Doppler, at each corner's DEM height), taking the
bounding (azimuth time, slant range) box, mapping it onto the swath axes, and
padding by a margin to guard processing edge effects (filter kernels,
coregistration search). A reference/secondary pair is cropped independently --
each acquisition's window comes from its own orbit. The crop is a plain integer
slice -- no resampling.

Cropped: ``swaths/zeroDopplerTime``, the ``frequency{A,B}`` images, their
``slantRange`` and ``validSamples``, and the radar-coordinate metadata grids
``metadata/geolocationGrid`` and ``metadata/processingInformation/parameters``
(``dopplerCentroid``, ``referenceTerrainHeight``), bracketed to span the cropped
swath. Restamped: ``identification/zeroDoppler{Start,End}Time`` and
``identification/boundingPolygon``. The GUNW writer copies these grids and
identification fields from the RSLC, so cropping them keeps the GUNW consistent.
Everything else (orbit, attitude, calibration, ...) is copied verbatim:
the workflow interpolates these coordinate-indexed tables, and the cropped grid
is a subset of their domain.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import isce3
import numpy as np
from nisar.products.readers import SLC


log = logging.getLogger(__name__)


def _band(h5: h5py.File) -> str:
    """Return the NISAR SAR band group under /science (LSAR for L-band, SSAR for S-band)."""
    band = next((b for b in h5['science'] if 'RSLC' in h5[f'science/{b}']), None)
    if band is None:
        raise ValueError(f'No RSLC band group found under /science in {h5.filename}')
    return band


def _copy_attrs(src: h5py.Group | h5py.Dataset, dst: h5py.Group | h5py.Dataset) -> None:
    """Copy all HDF5 attributes from ``src`` onto ``dst``."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _padded_slice(axis: np.ndarray, lo: float, hi: float, margin: int) -> tuple[int, int]:
    """Return a padded (start, stop) slice covering [lo, hi] on a strictly increasing axis."""
    if np.any(np.diff(axis) <= 0):
        raise ValueError('Axis must be strictly increasing for searchsorted to be valid.')
    if hi < axis[0] or lo > axis[-1]:
        raise ValueError(f'Requested extent {lo:.3f}-{hi:.3f} does not overlap the axis bounds.')
    i0, i1 = np.searchsorted(axis, [lo, hi])
    # Pad each side and clamp to the axis bounds.
    return int(max(0, i0 - margin)), int(min(len(axis), i1 + margin))


def _bracket(axis: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Return the (start, stop) slice bracketing [lo, hi] on an axis (one node outside each side)."""
    return max(0, int(np.searchsorted(axis, lo, 'right')) - 1), min(
        len(axis), int(np.searchsorted(axis, hi, 'left')) + 1
    )


def _solve_aoi_corners(
    orbit: isce3.core.Orbit,
    radar_grid: isce3.product.RadarGridParameters,
    bbox_wgs84: list[float],
    dem_file: str | Path,
) -> tuple[list[float], list[float]]:
    """geo2rdr the four AOI corners into the radar grid; return (az_times, slant_ranges).

    Each corner is taken at its DEM height (bilinear; the DEM's reference height
    outside its extent) and solved zero-Doppler, since NISAR RSLC is zero-Doppler
    processed. geo2rdr solves one point at a time, so the corners are looped.
    """
    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()  # empty LUT == Doppler 0 (zero-Doppler RSLC)
    dem_raster = isce3.io.Raster(str(dem_file))
    dem = isce3.geometry.DEMInterpolator(dem_raster)
    dem.load_dem(dem_raster)

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    corners = [(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_min), (lon_max, lat_max)]
    az_times, slant_ranges = [], []
    for lon, lat in corners:
        lon_rad, lat_rad = np.radians(lon), np.radians(lat)
        llh = np.array([lon_rad, lat_rad, dem.interpolate_lonlat(lon_rad, lat_rad)])
        try:
            aztime, srange = isce3.geometry.geo2rdr(
                llh, ellipsoid, orbit, zero_doppler, radar_grid.wavelength, radar_grid.lookside
            )
        except RuntimeError as e:
            # geo2rdr does not converge for a corner off the reachable geometry.
            raise ValueError(
                f'geo2rdr could not solve AOI corner (lon={lon}, lat={lat}); the AOI may not overlap this acquisition.'
            ) from e
        az_times.append(aztime)
        slant_ranges.append(srange)
    return az_times, slant_ranges


def _check_epoch_alignment(radar_grid: isce3.product.RadarGridParameters, swath_t: np.ndarray) -> None:
    """Fail if the radar-grid epoch disagrees with the swath time axis.

    geo2rdr azimuth times index into ``swath_t`` only if both count from the same
    epoch (``radar_grid.sensing_start == swath_t[0]`` for NISAR); tolerate under
    half an azimuth sample, beyond which every line index would be shifted.
    """
    az_spacing = swath_t[1] - swath_t[0]
    if abs(radar_grid.sensing_start - swath_t[0]) > 0.5 * az_spacing:
        raise RuntimeError(
            'RSLC radar-grid epoch does not match the swath zeroDopplerTime '
            'axis; geo2rdr azimuth times would be offset.'
        )


def aoi_to_radar_window(
    slc: SLC,
    bbox_wgs84: list[float],
    dem_file: str | Path,
    margin: int = 512,
) -> dict[str, tuple[int, int]]:
    """Find the radar-coordinate crop window for ``bbox_wgs84`` in one RSLC.

    Solves the four AOI corners into the radar grid with isce3 ``geo2rdr`` at
    their DEM heights, then maps the bounding (azimuth time, slant range) extent
    onto the swath axes. The window is independent per RSLC -- the AOI maps to
    different lines/pixels in each acquisition.

    Args:
        slc: Open NISAR RSLC reader.
        bbox_wgs84: AOI bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 degrees.
        dem_file: Path of a DEM GeoTIFF in WGS84 (lon/lat) used for corner heights.
        margin: Padding in frequencyA samples / azimuth lines added on every side.

    Returns:
        window: Dict of {'az': (a0, a1), 'frequencyA': (p0, p1), ...} index slices.
    """
    # frequencyA radar grid supplies the wavelength and look side geo2rdr needs
    # (geometry is shared across frequencies, and frequencyA is always present).
    radar_grid = slc.getRadarGrid('A')
    freq_groups = [f'frequency{x}' for x in slc.frequencies]
    az_times, slant_ranges = _solve_aoi_corners(slc.getOrbit(), radar_grid, bbox_wgs84, dem_file)

    # Read the stored swath axes (independent of the radar grid, so the epoch
    # check below is a real cross-check) and window each one.
    with h5py.File(slc.filename, 'r') as f:
        swaths = f'science/{_band(f)}/RSLC/swaths'
        swath_t = f[f'{swaths}/zeroDopplerTime'][()]
        slant_axes = {fr: f[f'{swaths}/{fr}/slantRange'][()] for fr in freq_groups}

    _check_epoch_alignment(radar_grid, swath_t)

    # Shared azimuth window, plus an independent range window per frequency (each
    # from its own slantRange axis covering the same physical AOI extent).
    a_lo, a_hi = min(az_times), max(az_times)
    r_lo, r_hi = min(slant_ranges), max(slant_ranges)
    window = {'az': _padded_slice(swath_t, a_lo, a_hi, margin)}
    window |= {fr: _padded_slice(slant_axes[fr], r_lo, r_hi, margin) for fr in freq_groups}

    a0, a1 = window['az']
    p0, p1 = window['frequencyA']
    log.info('Radar window %s: az %d lines, frequencyA %d px', Path(slc.filename).name, a1 - a0, p1 - p0)
    return window


def get_polarizations(slc: SLC) -> dict[str, list[str]]:
    """Return the polarizations per frequency group for an RSLC.

    Reads them from the NISAR product reader (spec-maintained upstream) so the
    HDF5 layout for frequencies/polarizations is not hard-coded here.

    Args:
        slc: Open NISAR RSLC reader.

    Returns:
        polarizations: Frequency group name -> polarization list (e.g. {'frequencyA': ['HH', 'HV']}).
    """
    return {f'frequency{x}': slc.polarizations[x] for x in slc.frequencies}


def _copy_cropped(src_ds: h5py.Dataset, dst_grp: h5py.Group, name: str, slices: tuple[slice, ...]) -> h5py.Dataset:
    """Create ``name`` in ``dst_grp`` from ``src_ds[slices]``, preserving storage layout and attrs."""
    data = src_ds[slices]
    # Preserve the source storage layout (chunked/compressed) so cropped images
    # stay compressed; clamp chunk dims to the (smaller) cropped shape.
    kwargs = {}
    if src_ds.chunks is not None:
        kwargs['chunks'] = tuple(min(c, s) for c, s in zip(src_ds.chunks, data.shape))
        kwargs['compression'] = src_ds.compression
        kwargs['compression_opts'] = src_ds.compression_opts
        kwargs['shuffle'] = src_ds.shuffle
    d = dst_grp.create_dataset(name, data=data, **kwargs)
    _copy_attrs(src_ds, d)
    return d


def _crop_grid_group(
    src_grp: h5py.Group, dst_grp: h5py.Group, az_lo: float, az_hi: float, rg_lo: float, rg_hi: float
) -> None:
    """Crop a radar-coordinate metadata grid group (geolocationGrid, processingInformation/parameters).

    The group has its own 1-D ``zeroDopplerTime``/``slantRange`` axes; data layers
    are indexed by them (1-D on whichever axis matches, or on their trailing two
    dims). Nested grid subgroups (e.g. ``frequency{A,B}``) are cropped recursively.
    Axes are *bracketed* (one node outside the swath extent each side) so the
    cropped grid still spans the swath.
    """
    grid_az = src_grp['zeroDopplerTime'][()]
    grid_rg = src_grp['slantRange'][()]
    n_az, n_rg = len(grid_az), len(grid_rg)
    gz0, gz1 = _bracket(grid_az, az_lo, az_hi)
    gs0, gs1 = _bracket(grid_rg, rg_lo, rg_hi)

    _copy_attrs(src_grp, dst_grp)
    for key, item in src_grp.items():
        if isinstance(item, h5py.Group):
            # A nested grid group (its own axes) is cropped recursively; anything
            # else (no axes) is copied verbatim.
            if 'zeroDopplerTime' in item and 'slantRange' in item:
                _crop_grid_group(item, dst_grp.create_group(key), az_lo, az_hi, rg_lo, rg_hi)
            else:
                src_grp.copy(key, dst_grp, name=key)
        elif key == 'zeroDopplerTime':
            _copy_cropped(item, dst_grp, key, (slice(gz0, gz1),))
        elif key == 'slantRange':
            _copy_cropped(item, dst_grp, key, (slice(gs0, gs1),))
        elif key == 'heightAboveEllipsoid':
            src_grp.copy(key, dst_grp, name=key)  # height axis -> not az/range indexed
        elif item.ndim >= 2 and item.shape[-2] == n_az and item.shape[-1] == n_rg:
            sl = (slice(None),) * (item.ndim - 2) + (slice(gz0, gz1), slice(gs0, gs1))
            _copy_cropped(item, dst_grp, key, sl)
        elif item.ndim == 1 and item.shape[0] == n_az:
            _copy_cropped(item, dst_grp, key, (slice(gz0, gz1),))  # az-indexed 1-D, e.g. referenceTerrainHeight
        elif item.ndim == 1 and item.shape[0] == n_rg:
            _copy_cropped(item, dst_grp, key, (slice(gs0, gs1),))
        else:
            # epsg, chirp weightings, run-config string, etc. -> not on the radar grid.
            src_grp.copy(key, dst_grp, name=key)


def _update_identification_times(
    dst: h5py.File, identification_path: str, units_attr: object, t0: float, t1: float
) -> None:
    """Restamp the identification start/end times to the cropped swath extent.

    ``t0``/``t1`` are the first/last cropped azimuth times (seconds since the epoch
    in ``units_attr``); otherwise the crop would report the full-scene times.
    """
    units = units_attr.decode() if isinstance(units_attr, bytes) else str(units_attr)
    if 'since' not in units:
        return
    epoch = datetime.fromisoformat(units.split('since', 1)[1].strip())
    for field, seconds in (('zeroDopplerStartTime', t0), ('zeroDopplerEndTime', t1)):
        path = f'{identification_path}/{field}'
        if path not in dst:
            continue
        value = (epoch + timedelta(seconds=float(seconds))).isoformat()
        sample = dst[path][()]
        dst[path][...] = value.encode() if isinstance(sample, bytes) else value


def _bounding_polygon_wkt(geoloc_grp: h5py.Group) -> str:
    """WKT footprint quad from a cropped geolocationGrid's corner lon/lat nodes.

    The grid's ``coordinateX``/``coordinateY`` are lon/lat (EPSG:4326); the four
    corner nodes at a mid height level trace the cropped swath outline.
    """
    lon = geoloc_grp['coordinateX'][()]  # (height, azimuth, range), degrees
    lat = geoloc_grp['coordinateY'][()]
    h = lon.shape[0] // 2  # representative height level
    corners = [(lon[h, a, r], lat[h, a, r]) for a, r in ((0, 0), (0, -1), (-1, -1), (-1, 0), (0, 0))]
    return 'POLYGON ((' + ', '.join(f'{x:.6f} {y:.6f}' for x, y in corners) + '))'


def _set_bounding_polygon(dst: h5py.File, identification_path: str, wkt: str) -> None:
    """Replace the (full-scene) identification boundingPolygon with the cropped footprint.

    The GUNW writer copies this field straight from the reference RSLC, so leaving
    the full-frame polygon would mislabel the cropped product's footprint.
    """
    path = f'{identification_path}/boundingPolygon'
    if path not in dst:
        return
    attrs = dict(dst[path].attrs)  # capture before delete; the dataset is resized
    del dst[path]
    d = dst.create_dataset(path, data=np.bytes_(wkt))
    for k, v in attrs.items():
        d.attrs[k] = v


def _mirror_except(src: h5py.Group, dst: h5py.Group, prune: set[str]) -> None:
    """Copy ``src`` into ``dst`` verbatim, skipping the ``prune`` subtrees (root-relative paths).

    Pruned objects are left for the caller to fill in cropped, but their parent
    groups are created. Everything else is copied wholesale in one HDF5 call,
    preserving attrs/dtype/fill value without reading big arrays into Python.
    """
    _copy_attrs(src, dst)
    for name, item in src.items():
        path = item.name.lstrip('/')
        if path in prune:
            continue  # caller fills this in with a cropped version
        if isinstance(item, h5py.Group) and any(p.startswith(f'{path}/') for p in prune):
            _mirror_except(item, dst.create_group(name), prune)  # an ancestor of a pruned path
        else:
            src.copy(name, dst, name=name)


def _crop_frequency_group(
    src_fr: h5py.Group, dst_fr: h5py.Group, a0: int, a1: int, p0: int, p1: int, pols: list[str]
) -> None:
    """Crop one ``frequency{A,B}`` subgroup: the pol images, ``slantRange``, and ``validSamples``."""
    _copy_attrs(src_fr, dst_fr)
    for name, item in src_fr.items():
        if name in pols:
            _copy_cropped(item, dst_fr, name, (slice(a0, a1), slice(p0, p1)))
        elif name == 'slantRange':
            _copy_cropped(item, dst_fr, name, (slice(p0, p1),))
        elif name.startswith('validSamples'):
            # Subset rows, shift the stored column indices into the cropped range
            # axis, and clip to the new sample range.
            vals = np.clip(item[a0:a1].astype(np.int64) - p0, 0, p1 - p0)
            d = dst_fr.create_dataset(name, data=vals.astype(item.dtype))
            _copy_attrs(item, d)
        else:
            src_fr.copy(name, dst_fr, name=name)  # slantRangeSpacing, listOfPolarizations, ...


def _crop_swaths_group(
    src_sw: h5py.Group, dst_sw: h5py.Group, window: dict[str, tuple[int, int]], polarizations: dict[str, list[str]]
) -> None:
    """Crop the ``swaths`` group: the shared azimuth axis and each ``frequency{A,B}`` subgroup."""
    _copy_attrs(src_sw, dst_sw)
    a0, a1 = window['az']
    for name, item in src_sw.items():
        if name.startswith('frequency'):
            p0, p1 = window[name]
            _crop_frequency_group(item, dst_sw.create_group(name), a0, a1, p0, p1, polarizations[name])
        elif name == 'zeroDopplerTime':
            _copy_cropped(item, dst_sw, name, (slice(a0, a1),))
        else:
            src_sw.copy(name, dst_sw, name=name)  # zeroDopplerTimeSpacing, ...


def crop_rslc(
    src_h5: str | Path,
    dst_h5: str | Path,
    window: dict[str, tuple[int, int]],
    polarizations: dict[str, list[str]],
) -> Path:
    """Write a cropped copy of an RSLC using a radar window.

    The whole product is mirrored verbatim except the two radar-coordinate
    subtrees (``swaths`` and ``geolocationGrid``), which are filled in cropped.
    Identification times are restamped to the cropped extent last.

    Args:
        src_h5: Path of the source RSLC h5 file.
        dst_h5: Path of the cropped RSLC h5 file to write.
        window: Index slices from :func:`aoi_to_radar_window`.
        polarizations: Frequency group name -> polarization list from :func:`get_polarizations`.

    Returns:
        dst_h5: Path of the written cropped RSLC.
    """
    src_h5, dst_h5 = Path(src_h5), Path(dst_h5)
    a0, a1 = window['az']
    p0a, p1a = window['frequencyA']

    with h5py.File(src_h5, 'r') as src, h5py.File(dst_h5, 'w') as dst:
        root = f'science/{_band(src)}'
        swaths = f'{root}/RSLC/swaths'
        geoloc = f'{root}/RSLC/metadata/geolocationGrid'
        params = f'{root}/RSLC/metadata/processingInformation/parameters'
        identification = f'{root}/identification'

        # Copy everything verbatim except the radar-coordinate subtrees, creating
        # their parent groups so the cropped versions can attach.
        _mirror_except(src, dst, {swaths, geoloc, params})

        # Cropped-swath extent, used to bracket the geolocation grid and restamp
        # the identification times.
        swath_t = src[f'{swaths}/zeroDopplerTime']
        slant_a = src[f'{swaths}/frequencyA/slantRange'][()]
        az_lo, az_hi = float(swath_t[a0]), float(swath_t[a1 - 1])
        rg_lo, rg_hi = float(slant_a[p0a]), float(slant_a[p1a - 1])

        # Fill the pruned subtrees with their cropped contents.
        _crop_swaths_group(src[swaths], dst.create_group(swaths), window, polarizations)
        _crop_grid_group(src[geoloc], dst.create_group(geoloc), az_lo, az_hi, rg_lo, rg_hi)
        _crop_grid_group(src[params], dst.create_group(params), az_lo, az_hi, rg_lo, rg_hi)

        # Restamp identification times and footprint to the cropped extent (the
        # GUNW writer copies both straight from the reference RSLC).
        time_units = swath_t.attrs.get('units')
        if time_units is not None:
            _update_identification_times(dst, identification, time_units, az_lo, az_hi)
        _set_bounding_polygon(dst, identification, _bounding_polygon_wkt(dst[geoloc]))
    log.info('Wrote cropped RSLC: %s', dst_h5)
    return dst_h5


def crop_rslc_pair(
    reference_rslc: str | Path,
    secondary_rslc: str | Path,
    bbox_wgs84: list[float],
    dem_file: str | Path,
    out_dir: str | Path,
    margin: int = 512,
) -> tuple[str, str]:
    """Crop a reference/secondary RSLC pair to the AOI radar window.

    Each file's window is solved independently from its own orbit (the AOI maps to
    different lines/pixels in each acquisition); ``margin`` guards processing edge
    effects (filter kernels, coregistration search).

    Args:
        reference_rslc: Path of the reference RSLC h5 file.
        secondary_rslc: Path of the secondary RSLC h5 file.
        bbox_wgs84: AOI bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 degrees.
        dem_file: Path of a DEM GeoTIFF in WGS84 used for corner heights.
        out_dir: Directory to write the cropped RSLCs into.
        margin: Padding in frequencyA samples / azimuth lines added on every side.

    Returns:
        paths: (ref_sub_path, sec_sub_path) of the cropped RSLCs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def crop_one(src: str | Path) -> str:
        slc = SLC(hdf5file=str(src))  # one reader per file, shared by both steps
        window = aoi_to_radar_window(slc, bbox_wgs84, dem_file, margin)
        dst = out_dir / f'{Path(src).stem}_sub.h5'
        crop_rslc(src, dst, window, get_polarizations(slc))
        return str(dst)

    return crop_one(reference_rslc), crop_one(secondary_rslc)
