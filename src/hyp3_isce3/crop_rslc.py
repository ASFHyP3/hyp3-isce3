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
coregistration search). The reference/secondary pair is cropped independently --
each acquisition's window comes from its own orbit. The crop is a plain integer
padded slice -- no resampling.

Cropped: only ``swaths`` -- ``zeroDopplerTime``, the ``frequency{A,B}`` images,
their ``slantRange`` and ``validSamples``. These image axes are what isce3 reads
to build the radar grid, so cropping them is what shrinks every radar-domain
step. Restamped: ``identification/zeroDoppler{Start,End}Time`` and
``identification/boundingPolygon`` (metadata the GUNW writer copies verbatim).

The window start is snapped down to a multiple of the InSAR multilook looks
(azimuth lines, range samples). The interferogram is multilooked from the
reference grid's first line/sample, so a crop whose origin is not on the look
grid averages a shifted set of pixels versus a full-frame run. That is invisible
in coherent ground (the look-cell average is smooth) but re-rolls the speckle
realization in decorrelated areas; snapping the origin makes the cropped product
match a full-frame run there too.

Everything else is copied verbatim: we crop only what the InSAR workflow consumes
from the swath images. The radar-coordinate metadata grids
``metadata/geolocationGrid`` and ``metadata/processingInformation/parameters``
(``dopplerCentroid``, ``referenceTerrainHeight``) are coordinate-indexed tables
the workflow interpolates, and the cropped swath is a subset of their domain, so a
full copy is valid -- the GUNW regenerates its geometry cube from the output
geocode grid and reads the radar grid from ``swaths``, so neither needs these
grids cropped today.

This is tied to which RSLC layers the current RSLC->GUNW workflow actually reads.
If a future processing change starts deriving GUNW layers from one of these grids,
or adds new radar-coordinate layers to the RSLC, this crop would need to be
extended to cover them.
"""

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import h5py
import isce3
import numpy as np
import yaml
from nisar.products.readers import SLC
from pyproj import Transformer


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


def _snap_to_grid(value: float, step: float, origin: float = 0, expand_up: bool = False) -> float:
    """Snap ``value`` onto the lattice ``origin + k * step``.

    Floors to the node at or below ``value`` (or ceils to the node at or above it when
    ``expand_up``). Both crops snap to a grid so the cropped output's cells coincide with a
    full-frame run's: the radar window to the multilook looks (``origin`` 0), the geocode box
    to the posting (``origin`` the runconfig anchor). Integer ``value``/``step`` give an int.
    """
    round_to_node = math.ceil if expand_up else math.floor
    return origin + round_to_node((value - origin) / step) * step


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
    az_looks: int = 16,
    rg_looks: int = 7,
) -> dict[str, tuple[int, int]]:
    """Find the radar-coordinate crop window for ``bbox_wgs84`` in one RSLC.

    Solves the four AOI corners into the radar grid with isce3 ``geo2rdr`` at
    their DEM heights, then maps the bounding (azimuth time, slant range) extent
    onto the swath axes. The window is independent per RSLC -- the AOI maps to
    different lines/pixels in each acquisition. The window start is snapped down to
    a multiple of the multilook looks so the crop's multilook cells line up with a
    full-frame run (see module docstring). An AOI that overruns the swath is clamped
    to the overlap with a warning.

    Args:
        slc: Open NISAR RSLC reader.
        bbox_wgs84: AOI bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 degrees.
        dem_file: Path of a DEM GeoTIFF in WGS84 (lon/lat) used for corner heights.
        margin: Padding in frequencyA samples / azimuth lines added on every side.
        az_looks: Azimuth multilook looks; the window start line is floored to a multiple of it.
        rg_looks: Range multilook looks; each frequency's window start sample is floored to a multiple of it.

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
    if (
        a_lo < swath_t[0]
        or a_hi > swath_t[-1]
        or any(r_lo < slant_axes[fr][0] or r_hi > slant_axes[fr][-1] for fr in freq_groups)
    ):
        log.warning('AOI extends past the swath of %s; cropping to the overlap.', Path(slc.filename).name)

    # Floor the window start to a multiple of the looks (multilook alignment; see module docstring).
    # rg_looks is the single crossmul range looks, applied to every frequency's own grid.
    a0, a1 = _padded_slice(swath_t, a_lo, a_hi, margin)
    window = {'az': (int(_snap_to_grid(a0, az_looks)), a1)}
    for fr in freq_groups:
        p0, p1 = _padded_slice(slant_axes[fr], r_lo, r_hi, margin)
        window[fr] = (int(_snap_to_grid(p0, rg_looks)), p1)

    a0, a1 = window['az']
    p0, p1 = window['frequencyA']
    log.info('Radar window %s: az %d lines, frequencyA %d px', Path(slc.filename).name, a1 - a0, p1 - p0)
    return window


def geocode_subset_box(
    subset: list[float], epsg_code: int, template_yaml: str | Path
) -> tuple[float, float, float, float]:
    """Reproject the WGS84 AOI to the output UTM box and snap it onto the full-frame geocode grid.

    Two steps that together produce the geocode output box. First reproject the lon/lat AOI
    into ``epsg_code`` meters. Then expand the box outward onto the runconfig's geocode lattice:
    isce3 posts every geocoded layer from the ``geocode.top_left`` anchor (each layer's first
    pixel center lands at ``anchor + posting/2``), so a full-frame run falls on a fixed lattice.
    A subset must share that lattice or its geocoded pixel centers sit a fraction of a pixel off
    the full-frame's, which re-rolls the speckle realization in decorrelated areas (coherent
    areas, being smooth, still match). Two runs coincide iff their input ``top_left`` are
    congruent modulo the coarsest posting, so we snap the corners to ``anchor + k * posting`` --
    finer layers, whose posting divides it, then align too. This is the geocode-grid analog of
    the radar-window snap in :func:`aoi_to_radar_window`; sourcing anchor/posting from the
    runconfig the workflow runs keeps them from drifting.

    Args:
        subset: AOI bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 degrees.
        epsg_code: Output UTM EPSG code; must match the template's geocode projection.
        template_yaml: Path of the downloaded JPL runconfig (from ``process.download_yaml``).

    Returns:
        subset_utm: (xmin, ymin, xmax, ymax) box in UTM meters, snapped to the geocode grid.
    """
    # Reproject the lon/lat rectangle to UTM; transform_bounds densifies the edges so the
    # returned box encloses it even where the boundary bulges between corners.
    lon_min, lat_min, lon_max, lat_max = subset
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_code}', always_xy=True)
    xmin, ymin, xmax, ymax = transformer.transform_bounds(lon_min, lat_min, lon_max, lat_max)

    geocode = yaml.safe_load(Path(template_yaml).read_text())['runconfig']['groups']['processing']['geocode']
    if int(geocode['output_epsg']) != epsg_code:
        log.warning(
            'Subset EPSG %d != template geocode EPSG %s; skipping grid snap.', epsg_code, geocode['output_epsg']
        )
        return xmin, ymin, xmax, ymax

    ax, ay = float(geocode['top_left']['x_abs']), float(geocode['top_left']['y_abs'])
    px = float(geocode['output_posting']['A']['x_posting'])  # coarsest posting; finer layers divide it
    py = float(geocode['output_posting']['A']['y_posting'])

    # Snap each corner to anchor + integer*posting, expanding outward to keep full AOI coverage.
    xmin = _snap_to_grid(xmin, px, ax)
    xmax = _snap_to_grid(xmax, px, ax, expand_up=True)
    ymin = _snap_to_grid(ymin, py, ay)
    ymax = _snap_to_grid(ymax, py, ay, expand_up=True)
    return xmin, ymin, xmax, ymax


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
        value = (epoch + timedelta(seconds=float(seconds))).isoformat(timespec='microseconds')
        sample = dst[path][()]
        dst[path][...] = value.encode() if isinstance(sample, bytes) else value


def _bounding_polygon_wkt(geoloc_grp: h5py.Group, az_lo: float, az_hi: float, rg_lo: float, rg_hi: float) -> str:
    """WKT footprint quad for the cropped swath, from the (full) geolocationGrid.

    The grid is copied verbatim, so its ``coordinateX``/``coordinateY`` (lon/lat,
    EPSG:4326) still span the whole frame. Window its own axes to one node outside
    the cropped (azimuth time, slant range) extent and trace the corner nodes at a
    mid height level to outline just the cropped swath.
    """
    a0, a1 = _padded_slice(geoloc_grp['zeroDopplerTime'][()], az_lo, az_hi, margin=1)
    r0, r1 = _padded_slice(geoloc_grp['slantRange'][()], rg_lo, rg_hi, margin=1)
    lon = geoloc_grp['coordinateX'][()]  # (height, azimuth, range), degrees
    lat = geoloc_grp['coordinateY'][()]
    h = lon.shape[0] // 2  # representative height level
    a1, r1 = a1 - 1, r1 - 1  # last in-range node indices
    corners = [(a0, r0), (a0, r1), (a1, r1), (a1, r0), (a0, r0)]
    pts = [(lon[h, a, r], lat[h, a, r]) for a, r in corners]
    return 'POLYGON ((' + ', '.join(f'{x:.6f} {y:.6f}' for x, y in pts) + '))'


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


def _copy_except(src: h5py.Group, dst: h5py.Group, skip: set[str]) -> None:
    """Copy ``src`` into ``dst`` verbatim, skipping the ``skip`` subtrees (root-relative paths).

    Skipped objects are left for the caller to fill in cropped, but their parent
    groups are created. Everything else is copied wholesale in one HDF5 call,
    preserving attrs/dtype/fill value without reading big arrays into Python.
    """
    _copy_attrs(src, dst)
    for name, item in src.items():
        path = item.name.lstrip('/')
        if path in skip:
            continue  # caller fills this in with a cropped version
        if isinstance(item, h5py.Group) and any(s.startswith(f'{path}/') for s in skip):
            _copy_except(item, dst.create_group(name), skip)  # an ancestor of a skipped path
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

    The whole product is copied verbatim except the ``swaths`` subtree (the image
    grids), which is filled in cropped. Identification times and the bounding
    polygon are restamped to the cropped extent last.

    Args:
        src_h5: Path of the source RSLC h5 file.
        dst_h5: Path of the cropped RSLC h5 file to write.
        window: Index slices from :func:`aoi_to_radar_window`.
        polarizations: Frequency group name -> polarization list from :func:`get_polarizations`.

    Returns:
        dst_h5: Path of the written cropped RSLC.
    """
    with h5py.File(Path(src_h5), 'r') as src:
        return crop_rslc_from_handle(src, dst_h5, window, polarizations)


def crop_rslc_from_handle(
    src: h5py.File,
    dst_h5: str | Path,
    window: dict[str, tuple[int, int]],
    polarizations: dict[str, list[str]],
) -> Path:
    """Write a cropped RSLC from an already-open source handle.

    Same crop as :func:`crop_rslc`, but from an open ``h5py.File`` rather than a path,
    so a remote handle's windowed slices become byte-range reads (see :func:`crop_streamed`)
    and the output is byte-for-byte identical to a local crop.

    Args:
        src: Open source RSLC, a local or remote ``h5py.File`` handle.
        dst_h5: Path of the cropped RSLC h5 file to write.
        window: Index slices from :func:`aoi_to_radar_window`.
        polarizations: Frequency group name -> polarization list from :func:`get_polarizations`.

    Returns:
        dst_h5: Path of the written cropped RSLC.
    """
    dst_h5 = Path(dst_h5)
    a0, a1 = window['az']
    p0a, p1a = window['frequencyA']

    with h5py.File(dst_h5, 'w') as dst:
        root = f'science/{_band(src)}'
        swaths = f'{root}/RSLC/swaths'
        geoloc = f'{root}/RSLC/metadata/geolocationGrid'
        identification = f'{root}/identification'

        # Copy everything verbatim except the swaths image subtree (filled in cropped
        # below); the metadata grids are left full -- see the module docstring.
        _copy_except(src, dst, {swaths})

        # Cropped-swath extent, used to outline the footprint and restamp the
        # identification times.
        swath_t = src[f'{swaths}/zeroDopplerTime']
        slant_a = src[f'{swaths}/frequencyA/slantRange'][()]
        az_lo, az_hi = float(swath_t[a0]), float(swath_t[a1 - 1])
        rg_lo, rg_hi = float(slant_a[p0a]), float(slant_a[p1a - 1])

        # Fill the pruned swaths subtree with its cropped contents.
        _crop_swaths_group(src[swaths], dst.create_group(swaths), window, polarizations)

        # Restamp identification times and footprint to the cropped extent (the
        # GUNW writer copies both straight from the reference RSLC).
        time_units = swath_t.attrs.get('units')
        if time_units is not None:
            _update_identification_times(dst, identification, time_units, az_lo, az_hi)
        _set_bounding_polygon(dst, identification, _bounding_polygon_wkt(src[geoloc], az_lo, az_hi, rg_lo, rg_hi))
    log.info('Wrote cropped RSLC: %s', dst_h5)
    return dst_h5


def write_skeleton(src: h5py.File, skeleton_h5: str | Path) -> Path:
    """Write a metadata-only copy of an RSLC: full structure, empty image datasets.

    The window solve and its isce3 readers open the product by path and read only
    metadata and the swath axes, never the pixels. Recreating the large pol images
    empty (same shape/dtype/layout) yields a tiny local file those path-based readers
    accept, so :func:`crop_streamed` can solve the window without downloading imagery.

    Args:
        src: Open source RSLC (typically a remote h5py handle).
        skeleton_h5: Path of the skeleton h5 file to write.

    Returns:
        skeleton_h5: Path of the written skeleton.
    """
    skeleton_h5 = Path(skeleton_h5)
    with h5py.File(skeleton_h5, 'w') as dst:
        swaths = f'science/{_band(src)}/RSLC/swaths'
        _copy_except(src, dst, {swaths})  # everything outside swaths is small metadata

        # Rebuild swaths: copy the axes/validSamples, but recreate the pol images empty.
        src_sw = src[swaths]
        dst_sw = dst.create_group(swaths)
        _copy_attrs(src_sw, dst_sw)
        for name, item in src_sw.items():
            if not name.startswith('frequency'):
                src_sw.copy(name, dst_sw, name=name)  # zeroDopplerTime axis, spacing, ...
                continue
            dst_fr = dst_sw.create_group(name)
            _copy_attrs(item, dst_fr)
            for sub, ds in item.items():
                # Recreate the large 2D complex pol images empty (shape only, no pixels).
                if isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.dtype.kind == 'c':
                    d = dst_fr.create_dataset(
                        sub, shape=ds.shape, dtype=ds.dtype, chunks=ds.chunks, compression=ds.compression
                    )
                    _copy_attrs(ds, d)
                else:
                    item.copy(sub, dst_fr, name=sub)  # slantRange, validSamples, ...
    log.info('Wrote RSLC skeleton: %s', skeleton_h5)
    return skeleton_h5


# --- streaming crop ---------------------------------------------------------
# Read an RSLC's AOI window over byte-range instead of downloading the whole product.
# RSLC imagery is chunked, compressed HDF5, so a windowed read pulls only the
# overlapping chunks. Per scene: stream a metadata-only skeleton (the path-based isce3
# readers need metadata, not pixels, to solve the window), then crop the window straight
# from the remote handle. Output is byte-for-byte identical to a local crop.

# CMR collection of the NISAR L1 RSLC granules (BETA, matching the rest of the pipeline).
RSLC_SHORT_NAME = 'NISAR_L1_RSLC_BETA_V1'
RSLC_SHORT_NAME_PROV = 'NISAR_L1_RSLC_PROVISIONAL_V1'


def open_remote_rslc(scene_name: str) -> h5py.File:
    """Open a NISAR RSLC over byte-range as an h5py handle (no full download)."""
    results = earthaccess.search_data(short_name=RSLC_SHORT_NAME, readable_granule_name=scene_name)
    if len(results) == 0:
        raise ValueError(f'No {RSLC_SHORT_NAME} granule found for {scene_name}')
    # earthaccess.open() handles auth + the S3/HTTPS redirect and block-caches reads.
    fileobj = earthaccess.open(results[:1])[0]
    return h5py.File(fileobj, 'r', driver='fileobj')


def stream_skeleton(scene_name: str) -> Path:
    """Stream a metadata-only skeleton of a remote RSLC to ``<scene>_skeleton.h5``.

    Stands in for the full product in the pre-crop steps; the name keeps the
    pipeline's name-parsing steps working.
    """
    with open_remote_rslc(scene_name) as remote:
        return write_skeleton(remote, f'{scene_name}_skeleton.h5')


def crop_streamed(
    scene_name: str,
    skeleton_path: str | Path,
    bbox_wgs84: list[float],
    dem_file: str | Path,
    margin: int = 512,
    az_looks: int = 16,
    rg_looks: int = 7,
) -> str:
    """Solve the radar window from the skeleton, then stream-crop the window to ``<scene>_sub.h5``.

    Reopens the remote product (cheap; avoids holding a handle across the intervening
    ancillary downloads) and reads only the windowed image chunks. ``bbox_wgs84`` is
    [lon_min, lat_min, lon_max, lat_max]; ``margin``/``az_looks``/``rg_looks`` pass through
    to :func:`aoi_to_radar_window`.
    """
    # Window + polarizations come from the local skeleton via the existing readers.
    slc = SLC(hdf5file=str(skeleton_path))
    try:
        window = aoi_to_radar_window(slc, bbox_wgs84, dem_file, margin, az_looks, rg_looks)
    except (ValueError, RuntimeError) as e:
        raise ValueError(f'AOI does not fit the RSLC {scene_name}: {e}') from e

    out_path = f'{scene_name}_sub.h5'
    with open_remote_rslc(scene_name) as remote:
        crop_rslc_from_handle(remote, out_path, window, get_polarizations(slc))
    return out_path
