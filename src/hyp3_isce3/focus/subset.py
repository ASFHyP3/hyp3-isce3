"""Focus only an AOI of an L0B: stream the pulses it needs, and keep the full file's statistics.

The AOI bounding box becomes the focus output grid (a zero-Doppler time span and a slant-range
span). isce3 then only decodes the pulses that grid needs, so the saving is in download and disk,
not compute: a skeleton of the L0B (everything but the signal arrays) is streamed, the pulse window
is planned on it with isce3's own block planning, and only those pulses of the signal arrays are
streamed into a cropped L0B.

A cropped L0B does not focus identically to the full one on its own: isce3 derives three things
from the whole file that change with the pulse set. The focus is therefore run with those taken
from the skeleton, which holds every pulse-indexed dataset except the signal:
- the raw grid (first pulse time and mean PRF), the lattice the dithered-PRF resample uses;
- the raw time span, which sets the orbit/attitude crop and the Doppler LUT lattice;
- the RX channel imbalance (caltone and LNA cal means over every pulse), used in beamforming.
With these, the cropped focus matches the full focus to the run-to-run floor.
"""

import contextlib
import logging
import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import earthaccess
import h5py
import isce3
import numpy as np
from nisar.antenna import pattern
from nisar.products.readers.Raw import open_rrsd
from nisar.products.readers.Raw.Raw import RawBase
from nisar.workflows import focus as nisar_focus

import hyp3_isce3
from hyp3_isce3.crop_rslc import get_multilook_looks, solve_aoi_corners


log = logging.getLogger(__name__)

# Pulses kept beyond the planned span, each side. The full file's raw grid can place its last lattice
# time a few pulses past the crop's last pulse, and the beamformer requires it inside the file.
PAD_SECONDS = 1.0
# Pulses copied per read of the signal arrays (chunked 512 pulses, so this reads whole chunk rows).
BLOCK_PULSES = 2048
# Lines/samples added around the AOI, so the GUNW workflow's crop window (its own 512 margin) fits inside.
MARGIN_SAMPLES = 512
# The GUNW runconfig whose multilook looks set the lattice a subset grid must start on (see snap_to_full_frame).
GUNW_FALLBACK = Path(hyp3_isce3.__file__).parent / 'schemas' / 'gunw_fallback.yaml'


def open_remote_l0b(granule: earthaccess.DataGranule) -> h5py.File:
    """Open an L0B granule's raw .h5 over HTTPS as an h5py handle (no download).

    Never walk this handle with visititems: an object visit makes HDF5 walk the signal arrays'
    chunk indexes, whose nodes are spread through the file, which streams the whole file.
    """
    name = granule['umm']['GranuleUR']
    # earthaccess.open handles auth and block-caches reads; the granule also carries a QA stats file.
    files = [f for f in earthaccess.open([granule]) if str(f.path).endswith(f'{name}.h5')]
    if not files:
        raise RuntimeError(f'No {name}.h5 among the granule files')
    return h5py.File(files[0], 'r', driver='fileobj')


def is_signal(dataset: h5py.Dataset) -> bool:
    """The BFPQ-encoded signal arrays: 2-D with a compound (r, i) dtype."""
    return dataset.ndim == 2 and dataset.dtype.fields is not None


def write_skeleton(src: h5py.File, skeleton_path: str | Path) -> Path:
    """Copy an L0B with the signal arrays left empty: full structure, every other dataset whole.

    The tree is walked by name (see open_remote_l0b). The signal arrays keep their shape and
    chunking, so the path-based isce3 readers accept the skeleton as an L0B.

    Args:
        src: Open source L0B (typically a remote handle).
        skeleton_path: Path of the skeleton to write.

    Returns:
        skeleton_path: Path of the written skeleton.
    """
    skeleton_path = Path(skeleton_path)
    with h5py.File(skeleton_path, 'w') as dst:
        dst.attrs.update(src.attrs)

        def copy_group(group: h5py.Group, path: str) -> None:
            for name in group:
                obj = group[name]
                child = f'{path}/{name}' if path else name
                if isinstance(obj, h5py.Group):
                    dst.require_group(child).attrs.update(obj.attrs)
                    copy_group(obj, child)
                elif is_signal(obj):
                    empty = dst.create_dataset(
                        child,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        chunks=obj.chunks,
                        compression=obj.compression,
                        compression_opts=obj.compression_opts,
                    )
                    empty.attrs.update(obj.attrs)
                else:
                    dst.copy(obj, child)

        copy_group(src, '')
    log.info(f'Wrote L0B skeleton: {skeleton_path} ({skeleton_path.stat().st_size / 1e6:.0f} MB)')
    return skeleton_path


@dataclass
class FocusGeometry:
    """What nisar.workflows.focus.focus() sets up before it reads any pulses, for one runconfig."""

    rawlist: list[RawBase]
    common_mode: object
    orbit: isce3.core.Orbit
    dem: isce3.geometry.DEMInterpolator
    fc_ref: float
    dop_ref: isce3.core.LUT2d
    range_spacings: dict[str, float]
    grid: isce3.product.RadarGridParameters  # the output grid of the reference band


def focus_geometry(cfg: object) -> FocusGeometry:
    """Mirror focus() up to its output grid, so planning and snapping use isce3's own grid.

    With the runconfig naming the skeleton (full pulse axis) and the template's frame times, the
    grid is the full-frame grid: the production RSLC's corner and extent.

    Args:
        cfg: The loaded runconfig's groups (nisar_focus.load_config(...).runconfig.groups).

    Returns:
        The geometry, with grid the reference-band output grid.
    """
    nf = nisar_focus
    rawnames = cfg.input_file_group.input_file_path
    rawlist = [open_rrsd(str(Path(name).resolve())) for name in rawnames]
    dem = nf.get_dem(cfg)
    common_mode = nf.get_common_mode(rawlist)

    grid_epoch, t0, t1, r0, r1 = nf.get_total_grid_bounds(rawnames)
    orbit = nf.get_orbit(cfg)
    attitude = nf.get_attitude(cfg)
    nf.require_ephemeris_overlap(orbit, t0, t1, 'Orbit')
    nf.require_ephemeris_overlap(attitude, t0, t1, 'Attitude')
    fc_ref, dop_ref = nf.make_doppler(cfg, epoch=grid_epoch, orbit=orbit, attitude=attitude, dem=dem)

    max_chirplen = nf.get_max_chirp_duration(cfg) * isce3.core.speed_of_light / 2
    range_spacings = nf.get_output_range_spacings(rawlist, common_mode)
    dr = min(range_spacings.values())
    max_prf = nf.get_max_prf(rawlist)
    side = nf.require_constant_look_side(rawlist)
    grid = nf.make_output_grid(
        cfg, grid_epoch, t0, t1, max_prf, r0, r1, dr, side, orbit, fc_ref, dop_ref, max_chirplen, dem
    )
    return FocusGeometry(rawlist, common_mode, orbit, dem, fc_ref, dop_ref, range_spacings, grid)


def snap_to_full_frame(
    t_lo: float,
    t_hi: float,
    r_lo: float,
    r_hi: float,
    grid: isce3.product.RadarGridParameters,
    az_block: int,
    rg_block: int,
) -> dict[str, datetime | float]:
    """Round a time/range span outward onto multiples of the multilook blocks from the full-frame corner.

    The GUNW workflow multilooks from its input RSLC's first line and sample, so a subset RSLC gives
    the same multilook cells as a full-frame one only if its corner is a whole number of blocks from
    the full-frame corner. The span is also cut to the full-frame grid.

    Args:
        t_lo: Earliest zero-Doppler time (s since grid.ref_epoch).
        t_hi: Latest zero-Doppler time (s since grid.ref_epoch).
        r_lo: Nearest slant range (m).
        r_hi: Farthest slant range (m).
        grid: The full-frame output grid.
        az_block: Azimuth block, in lines.
        rg_block: Range block, in samples of grid.

    Returns:
        The output_grid runconfig fields: start_time, end_time, start_range, end_range.
    """
    lines = (t_lo - grid.sensing_start) * grid.prf, (t_hi - grid.sensing_start) * grid.prf
    samples = (
        (r_lo - grid.starting_range) / grid.range_pixel_spacing,
        (r_hi - grid.starting_range) / grid.range_pixel_spacing,
    )
    i_lo = max(0, math.floor(lines[0] / az_block) * az_block)
    i_hi = min(grid.length, math.ceil(lines[1] / az_block) * az_block)
    j_lo = max(0, math.floor(samples[0] / rg_block) * rg_block)
    j_hi = min(grid.width, math.ceil(samples[1] / rg_block) * rg_block)
    if i_lo >= i_hi or j_lo >= j_hi:
        raise ValueError('The AOI does not overlap the frame')
    if (i_lo, i_hi, j_lo, j_hi) != (
        lines[0] // az_block * az_block,
        -(-lines[1] // az_block) * az_block,
        samples[0] // rg_block * rg_block,
        -(-samples[1] // rg_block) * rg_block,
    ):
        log.warning('The AOI extends past the frame; focusing the overlap.')
    epoch = datetime.fromisoformat(grid.ref_epoch.isoformat()[:26]).replace(tzinfo=UTC)
    return {
        'start_time': epoch + timedelta(seconds=grid.sensing_start + i_lo / grid.prf),
        'end_time': epoch + timedelta(seconds=grid.sensing_start + i_hi / grid.prf),
        'start_range': grid.starting_range + j_lo * grid.range_pixel_spacing,
        'end_range': grid.starting_range + j_hi * grid.range_pixel_spacing,
    }


def microsecond_exact_lines(prf: float) -> int:
    """The line count whose multiples are whole microseconds at this PRF (PRF / gcd(PRF, 1e6)).

    Args:
        prf: Output PRF in Hz; the lattice argument only holds for a whole number.

    Returns:
        The smallest k with k / prf a whole number of microseconds.
    """
    if prf != int(prf):
        raise ValueError(f'output_prf {prf} is not a whole number of Hz')
    return int(prf) // math.gcd(int(prf), 10**6)


def output_grid_for_aoi(cfg: object, subset: list[float]) -> dict[str, datetime | float]:
    """Map an AOI onto the focus output grid: its zero-Doppler time and slant-range spans.

    The four AOI corners are solved into radar coordinates with geo2rdr at their DEM heights (the
    GUNW crop's corner solve), padded by MARGIN_SAMPLES, and snapped outward onto the full-frame
    grid's multilook lattice (snap_to_full_frame), with the blocks from the GUNW runconfig's looks.

    Args:
        cfg: The loaded runconfig's groups, naming the skeleton and carrying the frame's output grid.
        subset: AOI bounding box [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX] in WGS84 degrees.

    Returns:
        The output_grid fields to set: start_time, end_time, start_range, end_range.
    """
    geometry = focus_geometry(cfg)
    grid = geometry.grid
    az_times, slant_ranges = solve_aoi_corners(geometry.orbit, grid, subset, cfg.dynamic_ancillary_file_group.dem_file)

    # Blocks: the lcm of the GUNW looks on every band, in this grid's lines and samples. Frequency B's
    # samples are every rskip-th of the reference band's, so its range looks count rskip samples each.
    az_step, rg_step, rg_looks_b = get_multilook_looks(GUNW_FALLBACK)
    dr = grid.range_pixel_spacing
    rskips = [int(np.round(spacing / dr)) for spacing in geometry.range_spacings.values()]
    rg_block = math.lcm(rg_step, *(rg_looks_b * rskip for rskip in rskips if rskip > 1))
    # The runconfig holds the start time to the microsecond, and k/PRF s is a whole number of
    # microseconds only when k is a multiple of PRF / gcd(PRF, 1e6) (19 lines at 1520 Hz). Folding
    # that into the block keeps every subset start exactly on the full-frame lattice.
    az_block = math.lcm(az_step, microsecond_exact_lines(grid.prf))

    t_lo, t_hi = min(az_times) - MARGIN_SAMPLES / grid.prf, max(az_times) + MARGIN_SAMPLES / grid.prf
    r_lo, r_hi = min(slant_ranges) - MARGIN_SAMPLES * dr, max(slant_ranges) + MARGIN_SAMPLES * dr
    out = snap_to_full_frame(t_lo, t_hi, r_lo, r_hi, grid, az_block, rg_block)
    log.info(
        f'AOI output grid: {out["start_time"]} to {out["end_time"]}, {out["start_range"]:.1f} to {out["end_range"]:.1f} m '
        f'(blocks of {az_block} lines, {rg_block} samples from the full-frame corner)'
    )
    return out


def needed_raw_span(cfg: object) -> tuple[isce3.core.DateTime, float, float]:
    """The raw time span the focus will read for cfg's output grid, from isce3's own block planning.

    Mirrors nisar.workflows.focus.focus() through plan_processing_blocks (zero-to-native Doppler
    reskew of every block corner plus the synthetic aperture, padded 10%) and total_bounds; the
    focus then picks pulses by bisecting the pulse times against this span.

    Args:
        cfg: The loaded runconfig's groups (nisar_focus.load_config(...).runconfig.groups).

    Returns:
        epoch: Reference epoch of the span.
        begin: Start of the span (s since epoch).
        end: End of the span (s since epoch).
    """
    nf = nisar_focus
    geometry = focus_geometry(cfg)
    dr = min(geometry.range_spacings.values())
    # Per-frequency output grid and Doppler, then the raw bounds of every azcomp block.
    bounds = []
    for frequency, band in nf.get_bands(geometry.common_mode).items():
        rskip = int(np.round(geometry.range_spacings[frequency] / dr))
        ogrid = geometry.grid[:, ::rskip]
        ogrid.wavelength = isce3.core.speed_of_light / band.center
        dop = nf.scale_doppler(geometry.dop_ref, band.center / geometry.fc_ref)
        bounds += nf.plan_processing_blocks(cfg, ogrid, dop, geometry.dem, geometry.orbit)
    begin, end = nf.total_bounds(bounds)
    log.info(f'The focus needs raw data in [{begin}, {end}] s since {geometry.grid.ref_epoch}')
    return geometry.grid.ref_epoch, begin, end


def pulse_windows(l0b_path: str, epoch: isce3.core.DateTime, begin: float, end: float) -> dict[str, tuple[int, int]]:
    """Pulse index window [start, stop) per transmit polarization covering [begin, end], as the focus picks it.

    Args:
        l0b_path: Path of the L0B (or its skeleton).
        epoch: Reference epoch of begin and end.
        begin: Start of the time span (s since epoch).
        end: End of the time span (s since epoch).

    Returns:
        {tx: (start, stop)} for every transmit polarization in the file.
    """
    raw = open_rrsd(l0b_path)
    windows = {}
    for frequency in raw.frequencies:
        for tx in {pol[0] for pol in raw.polarizations[frequency]}:
            times, _ = raw.getRadarGrid(frequency, tx=tx, epoch=epoch)
            window = (bisect_left(times, begin), bisect_right(times, end))
            # Frequencies A and B share one pulse axis per transmit, so the windows must agree.
            if windows.setdefault(tx, window) != window:
                raise ValueError(f'tx{tx} pulse windows differ between frequencies: {windows[tx]} vs {window}')
            log.info(f'frequency{frequency} tx{tx}: pulses [{window[0]}, {window[1]}] of {len(times)}')
    return windows


def crop_dataset(src: h5py.Dataset, dst: h5py.File, path: str, start: int, stop: int) -> None:
    """Write src[start:stop] to dst at path with the same layout, in pulse blocks for the big arrays."""
    shape = (stop - start, *src.shape[1:])
    chunks = tuple(min(c, s) for c, s in zip(src.chunks, shape)) if src.chunks else None
    out = dst.create_dataset(
        path,
        shape=shape,
        dtype=src.dtype,
        chunks=chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
    )
    out.attrs.update(src.attrs)
    for i in range(start, stop, BLOCK_PULSES):
        j = min(i + BLOCK_PULSES, stop)
        out[i - start : j - start] = src[i:j]


def crop_l0b(
    skeleton_path: str | Path, crop_path: str | Path, windows: dict[str, tuple[int, int]], signal_src: h5py.File
) -> Path:
    """Write a cropped L0B: the skeleton with every pulse-indexed dataset sliced, signal windows from signal_src.

    A dataset is pulse-indexed when its first axis has its transmit group's pulse count; these live
    under swaths and highRateTelemetry. Everything else is copied unchanged. The identification
    times follow the kept pulses; the bounding polygon is left as the full L0B's.

    Args:
        skeleton_path: Path of the skeleton (signal arrays empty).
        crop_path: Path of the cropped L0B to write.
        windows: {tx: (start, stop)} pulse windows from pulse_windows.
        signal_src: Handle the signal arrays are read from (typically the remote L0B).

    Returns:
        crop_path: Path of the written crop.
    """
    crop_path = Path(crop_path)
    with h5py.File(skeleton_path, 'r') as src, h5py.File(crop_path, 'w') as dst:
        band = next(iter(src['science']))
        swaths = src[f'science/{band}/RRSD/swaths']
        first_freq = next(k for k in swaths if k.startswith('frequency'))
        # Pulse count per transmit, from the UTCtime axis (shared by every frequency).
        n_pulses = {tx: len(swaths[f'{first_freq}/tx{tx}/UTCtime']) for tx in windows}
        dst.attrs.update(src.attrs)
        sliced = 0

        def visit(path: str, obj: h5py.Group | h5py.Dataset) -> None:
            nonlocal sliced
            if isinstance(obj, h5py.Group):
                dst.require_group(path).attrs.update(obj.attrs)
                return
            tx = next((tx for tx in windows if f'/tx{tx}/' in f'/{path}/'), None)
            if tx is not None and obj.ndim >= 1 and obj.shape[0] == n_pulses[tx]:
                crop_dataset(signal_src[path] if is_signal(obj) else obj, dst, path, *windows[tx])
                sliced += 1
            else:
                dst.copy(obj, path)

        src.visititems(visit)  # the skeleton is local and its signal arrays are empty, so a visit is cheap

        # Identification times follow the kept pulses; the polygon is left as the full L0B's.
        ident = dst[f'science/{band}/identification']
        utc = dst[f'science/{band}/RRSD/swaths/{first_freq}/tx{next(iter(windows))}/UTCtime']
        epoch = isce3.io.get_ref_epoch(utc.parent, 'UTCtime')
        for name, seconds in (('zeroDopplerStartTime', utc[0]), ('zeroDopplerEndTime', utc[-1])):
            old = ident[name][()].decode()
            ident[name][()] = (epoch + isce3.core.TimeDelta(float(seconds))).isoformat()[: len(old)].encode()
    log.info(f'Wrote cropped L0B: {crop_path} ({crop_path.stat().st_size / 1e6:.0f} MB, {sliced} datasets sliced)')
    return crop_path


def crop_streamed_l0b(granule: earthaccess.DataGranule, skeleton_path: str, runconfig_path: str | Path) -> str:
    """Plan the pulse window the runconfig's output grid needs and stream it into <granule>_crop.h5.

    Args:
        granule: The L0B granule to stream the signal from.
        skeleton_path: Path of its skeleton, which the runconfig's input_file_path must name.
        runconfig_path: Path of the focus runconfig with the output grid set.

    Returns:
        crop_path: Path of the cropped L0B.
    """
    cfg = nisar_focus.load_config(str(runconfig_path)).runconfig.groups
    epoch, begin, end = needed_raw_span(cfg)
    windows = pulse_windows(skeleton_path, epoch, begin - PAD_SECONDS, end + PAD_SECONDS)
    crop_path = f'{granule["umm"]["GranuleUR"]}_crop.h5'
    with open_remote_l0b(granule) as remote:
        crop_l0b(skeleton_path, crop_path, windows, remote)
    return crop_path


@contextlib.contextmanager
def full_file_quantities(skeleton_path: str | None) -> Iterator[None]:
    """Run the focus with the raw grid, raw span, and RX channel imbalance taken from the skeleton.

    These are the three things isce3 derives from the whole file (see the module docstring). The
    reader and workflow functions that compute them are redirected to the skeleton for any other
    file, and restored afterwards. The raw grid is the skeleton's grid sliced to the crop's pulses,
    so the crop must be a contiguous slice of the skeleton's pulse axis.

    Args:
        skeleton_path: Path of the skeleton; None leaves the focus unchanged.
    """
    if skeleton_path is None:
        yield
        return

    skeleton = Path(skeleton_path).resolve()
    full = None

    def get_full() -> RawBase:
        nonlocal full
        if full is None:
            full = open_rrsd(str(skeleton))
        return full

    orig_grid, orig_prf = RawBase.getRadarGrid, RawBase.getNominalPRF
    orig_bounds = nisar_focus.get_total_grid_bounds
    orig_imbalance = pattern.compute_all_rx_channel_imbalances_from_l0b

    def getRadarGrid(
        self: RawBase,
        frequency: str = 'A',
        tx: str = 'H',
        prf: float | None = None,
        epoch: isce3.core.DateTime | None = None,
    ) -> tuple:  # noqa: N802
        t, grid = orig_grid(self, frequency, tx, prf, epoch)
        if prf or Path(self.filename).resolve() == skeleton:
            return t, grid
        full_t, full_grid = orig_grid(get_full(), frequency, tx, None, epoch)
        i0 = int(np.searchsorted(full_t, t[0]))
        if not np.allclose(full_t[i0 : i0 + len(t)], t, atol=1e-9):
            raise ValueError(f'{self.filename} is not a contiguous pulse slice of {skeleton.name}')
        return t, full_grid[i0 : i0 + len(t), :]

    def getNominalPRF(self: RawBase, frequency: str = 'A', tx: str = 'H') -> float:  # noqa: N802
        return orig_prf(get_full(), frequency, tx)

    def get_total_grid_bounds(rawfiles: list[str]) -> tuple:
        return orig_bounds([str(skeleton)])

    def compute_all_rx_channel_imbalances_from_l0b(raw: RawBase, **kwargs: object) -> dict:
        return orig_imbalance(get_full(), **kwargs)

    RawBase.getRadarGrid, RawBase.getNominalPRF = getRadarGrid, getNominalPRF
    nisar_focus.get_total_grid_bounds = get_total_grid_bounds
    pattern.compute_all_rx_channel_imbalances_from_l0b = compute_all_rx_channel_imbalances_from_l0b
    try:
        yield
    finally:
        RawBase.getRadarGrid, RawBase.getNominalPRF = orig_grid, orig_prf
        nisar_focus.get_total_grid_bounds = orig_bounds
        pattern.compute_all_rx_channel_imbalances_from_l0b = orig_imbalance
