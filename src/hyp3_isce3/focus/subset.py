"""Focus only an AOI of an L0B: stream the pulses it needs into a skeleton of the L0B, and focus that.

The AOI bounding box becomes the focus output grid, snapped onto the full-frame grid's multilook
lattice. A skeleton of the L0B is streamed: every dataset whole, with the signal arrays allocated
but empty. Only the pulses the output grid needs are then streamed into its signal arrays. isce3
reads signal only inside that pulse window, and everything it derives from the whole file (raw grid,
raw time span, RX channel imbalance) comes from the skeleton's full-length datasets, so the focus
matches a full-L0B focus on the same grid.

To match the production RSLC, the pulse window is also widened so the tone-rank RFI blocks tile
the same pulses as the production run's (see align_rfi_blocks). A residual remains: isce3's
tone-rank step does not clear its block buffer between blocks, so the partial range windows carry
tails from earlier blocks that a subset run does not have.
"""

import contextlib
import logging
import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import earthaccess
import h5py
import isce3
import numpy as np
from nisar.products.readers.Raw import open_rrsd
from nisar.workflows import focus as nisar_focus

from hyp3_isce3.crop_rslc import _snap_to_grid, get_multilook_looks, solve_aoi_corners


log = logging.getLogger(__name__)

# Pulses copied per read of the signal arrays (chunked 512 pulses, so this reads whole chunk rows).
BLOCK_PULSES = 2048
# Lines/samples added around the AOI, so the GUNW workflow's crop window (its own 512 margin) fits inside.
MARGIN_SAMPLES = 512


class OutputGrid(TypedDict):
    """The focus runconfig's output_grid fields a subset sets."""

    start_time: datetime
    end_time: datetime
    start_range: float
    end_range: float


def to_datetime(epoch: isce3.core.DateTime) -> datetime:
    """An isce3 DateTime as a UTC datetime, to the microsecond."""
    return datetime.fromisoformat(epoch.isoformat()[:26]).replace(tzinfo=UTC)


@contextlib.contextmanager
def open_remote_l0b(granule: earthaccess.DataGranule) -> Iterator[h5py.File]:
    """Open an L0B granule's raw .h5 over HTTPS as an h5py handle (no download), closing every granule file after.

    Never walk this handle with visititems: an object visit makes HDF5 walk the signal arrays'
    chunk indexes, whose nodes are spread through the file, which streams the whole file.
    """
    name = granule['umm']['GranuleUR']
    # earthaccess.open handles auth and block-caches reads; the granule also carries a QA stats file.
    files = earthaccess.open([granule])
    try:
        h5_files = [f for f in files if str(f.path).endswith(f'{name}.h5')]
        if not h5_files:
            raise RuntimeError(f'No {name}.h5 among the granule files')
        # Closing the h5py handle does not close the remote file under it, so close each one here.
        with h5py.File(h5_files[0], 'r', driver='fileobj') as h5:
            yield h5
    finally:
        for f in files:
            f.close()


def is_signal(dataset: h5py.Dataset) -> bool:
    """The BFPQ-encoded signal arrays: 2-D with a compound (r, i) dtype."""
    return dataset.ndim == 2 and dataset.dtype.fields is not None


def write_skeleton(src: h5py.File, skeleton_path: str | Path) -> Path:
    """Copy an L0B with the signal arrays left empty: full structure, every other dataset whole.

    The tree is walked by name (see open_remote_l0b). The signal arrays keep their shape and
    chunking, so isce3 reads the skeleton as the L0B once fill_pulses has written its pulse window.

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

    common_mode: object
    orbit: isce3.core.Orbit
    dem: isce3.geometry.DEMInterpolator
    fc_ref: float
    dop_ref: isce3.core.LUT2d
    range_spacings: dict[str, float]
    grid: isce3.product.RadarGridParameters  # the output grid of the reference band


def focus_geometry(cfg: nisar_focus.Struct) -> FocusGeometry:
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
    return FocusGeometry(common_mode, orbit, dem, fc_ref, dop_ref, range_spacings, grid)


def snap_to_full_frame(
    t_lo: float,
    t_hi: float,
    r_lo: float,
    r_hi: float,
    grid: isce3.product.RadarGridParameters,
    az_block: int,
    rg_block: int,
) -> OutputGrid:
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
    snapped = (
        _snap_to_grid(lines[0], az_block),
        _snap_to_grid(lines[1], az_block, expand_up=True),
        _snap_to_grid(samples[0], rg_block),
        _snap_to_grid(samples[1], rg_block, expand_up=True),
    )
    i_lo, i_hi = max(0, snapped[0]), min(grid.length, snapped[1])
    j_lo, j_hi = max(0, snapped[2]), min(grid.width, snapped[3])
    if i_lo >= i_hi or j_lo >= j_hi:
        raise ValueError('The AOI does not overlap the frame')
    if (i_lo, i_hi, j_lo, j_hi) != snapped:
        log.warning('The AOI extends past the frame; focusing the overlap.')
    epoch = to_datetime(grid.ref_epoch)
    return {
        'start_time': epoch + timedelta(seconds=grid.sensing_start + i_lo / grid.prf),
        'end_time': epoch + timedelta(seconds=grid.sensing_start + i_hi / grid.prf),
        'start_range': grid.starting_range + j_lo * grid.range_pixel_spacing,
        'end_range': grid.starting_range + j_hi * grid.range_pixel_spacing,
    }


def output_grid_for_aoi(
    cfg: nisar_focus.Struct, subset: list[float], geometry: FocusGeometry, gunw_yaml: Path
) -> OutputGrid:
    """Map an AOI onto the focus output grid: its zero-Doppler time and slant-range spans.

    The four AOI corners are solved into radar coordinates with geo2rdr at their DEM heights (the
    GUNW crop's corner solve), padded by MARGIN_SAMPLES, and snapped outward onto the full-frame
    grid's multilook lattice (snap_to_full_frame), with the blocks from the frame's GUNW runconfig looks.

    Args:
        cfg: The loaded runconfig's groups, naming the skeleton and carrying the frame's output grid.
        subset: AOI bounding box [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX] in WGS84 degrees.
        geometry: focus_geometry(cfg).
        gunw_yaml: The GUNW runconfig the GUNW workflow will use on this frame (process.download_yaml), whose
            looks set the lattice; production frames differ in range looks by mode.

    Returns:
        The output_grid fields to set: start_time, end_time, start_range, end_range.
    """
    grid = geometry.grid
    az_times, slant_ranges = solve_aoi_corners(geometry.orbit, grid, subset, cfg.dynamic_ancillary_file_group.dem_file)

    # Blocks: the lcm of the GUNW looks on every band, in this grid's lines and samples. Frequency B's
    # samples are every rskip-th of the reference band's, so its range looks count rskip samples each.
    az_step, rg_step, rg_looks_b = get_multilook_looks(gunw_yaml)
    dr = grid.range_pixel_spacing
    rskips = [int(np.round(spacing / dr)) for spacing in geometry.range_spacings.values()]
    rg_block = math.lcm(rg_step, *(rg_looks_b * rskip for rskip in rskips if rskip > 1))
    # The runconfig holds the start time to the microsecond, and k/PRF s is a whole number of
    # microseconds only when k is a multiple of PRF / gcd(PRF, 1e6) (19 lines at 1520 Hz). Folding
    # that into the block keeps every subset start exactly on the full-frame lattice.
    if grid.prf != int(grid.prf):
        raise ValueError(f'output_prf {grid.prf} is not a whole number of Hz')
    az_block = math.lcm(az_step, int(grid.prf) // math.gcd(int(grid.prf), 10**6))

    t_lo, t_hi = min(az_times) - MARGIN_SAMPLES / grid.prf, max(az_times) + MARGIN_SAMPLES / grid.prf
    r_lo, r_hi = min(slant_ranges) - MARGIN_SAMPLES * dr, max(slant_ranges) + MARGIN_SAMPLES * dr
    out = snap_to_full_frame(t_lo, t_hi, r_lo, r_hi, grid, az_block, rg_block)
    log.info(
        f'AOI output grid: {out["start_time"]} to {out["end_time"]}, {out["start_range"]:.1f} to {out["end_range"]:.1f} m '
        f'(blocks of {az_block} lines, {rg_block} samples from the full-frame corner)'
    )
    return out


def needed_raw_span(cfg: nisar_focus.Struct, geometry: FocusGeometry) -> tuple[isce3.core.DateTime, float, float]:
    """The raw time span the focus will read for cfg's output grid, from isce3's own block planning.

    Mirrors nisar.workflows.focus.focus() through plan_processing_blocks (zero-to-native Doppler
    reskew of every block corner plus the synthetic aperture, padded 10%) and total_bounds; the
    focus then picks pulses by bisecting the pulse times against this span.

    Args:
        cfg: The loaded runconfig's groups (nisar_focus.load_config(...).runconfig.groups).
        geometry: focus_geometry(cfg).

    Returns:
        epoch: Reference epoch of the span.
        begin: Start of the span (s since epoch).
        end: End of the span (s since epoch).
    """
    nf = nisar_focus
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
    windows: dict[str, tuple[int, int]] = {}
    for frequency in raw.frequencies:
        for tx in {pol[0] for pol in raw.polarizations[frequency]}:
            times, _ = raw.getRadarGrid(frequency, tx=tx, epoch=epoch)
            window = (bisect_left(times, begin), bisect_right(times, end))
            # Frequencies A and B share one pulse axis per transmit, so the windows must agree.
            if windows.setdefault(tx, window) != window:
                raise ValueError(f'tx{tx} pulse windows differ between frequencies: {windows[tx]} vs {window}')
            log.info(f'frequency{frequency} tx{tx}: pulses [{window[0]}, {window[1]}] of {len(times)}')
    return windows


@dataclass
class RfiBlocks:
    """The tone-rank RFI blocks of the full-frame (production) focus, on one transmit polarization's pulses."""

    tx: str
    times: np.ndarray  # pulse times of the full file, s since the focus's grid epoch
    start: int  # first pulse the full-frame focus reads
    stop: int  # one past the last
    size: int  # pulses per block


def production_rfi_blocks(
    cfg: nisar_focus.Struct, geometry: FocusGeometry, skeleton_path: str, polarizations: list[str] | None
) -> RfiBlocks | None:
    """The tone-rank RFI blocks of a full-frame focus, or None when the runconfig does not mitigate with tone-rank.

    Args:
        cfg: The loaded runconfig's groups, naming the skeleton and carrying the frame's output grid.
        geometry: focus_geometry(cfg).
        skeleton_path: Path of the skeleton.
        polarizations: The polarizations to focus; None focuses every channel.
    """
    rfi = cfg.processing.radio_frequency_interference
    # Only tone-rank mitigation changes the pulses block by block (detection alone leaves them as read).
    if not (rfi.detection_enabled and rfi.mitigation_enabled and rfi.mitigation_algorithm.lower() == 'tone-rank'):
        return None
    raw = open_rrsd(skeleton_path)
    txs = {
        pol[0]
        for frequency in raw.frequencies
        for pol in raw.polarizations[frequency]
        if not polarizations or pol in polarizations
    }
    # Each transmit has its own pulse axis and so its own blocks, but focus reads one span for all.
    if len(txs) != 1:
        raise NotImplementedError(
            f'RFI block alignment needs one transmit polarization; focusing tx {sorted(txs)}. '
            'Pass --polarizations with a single transmit (e.g. HH HV) to subset this L0B.'
        )
    (tx,) = txs
    epoch, begin, end = needed_raw_span(cfg, geometry)
    times, _ = raw.getRadarGrid(raw.frequencies[0], tx=tx, epoch=epoch)
    times = np.asarray(times)
    blocks = RfiBlocks(tx, times, bisect_left(times, begin), bisect_right(times, end), rfi.tone_rank.block_dims[0])
    log.info(f'Full-frame focus reads tx{tx} pulses [{blocks.start}, {blocks.stop}) in RFI blocks of {blocks.size}')
    return blocks


def align_rfi_blocks(blocks: RfiBlocks, begin: float, end: float) -> tuple[float, float]:
    """Widen the raw span [begin, end] so the pulses read tile into the same RFI blocks as the production run.

    The tone-rank step cuts the pulses read into blocks of blocks.size from the first pulse read, and
    slides its last block back to end at the last pulse (isce3.noise.noise_power_est_func.cpi_slice_gen).
    The window starts a whole number of blocks after the production run's first pulse and holds a
    whole number of blocks; if it reaches into the production run's slid last block it ends where the
    production run does, so its own last block slides back onto the same pulses. A span reaching past
    the production run's pulses is cut to them: production focused without those pulses too.

    Args:
        blocks: The production run's RFI blocks.
        begin: Start of the span the focus plans to read (s since the blocks' epoch).
        end: End of that span (s).

    Returns:
        The widened span, between pulse times so a bisection picks exactly the aligned pulses.
    """
    t, size = blocks.times, blocks.size
    first, last = bisect_left(t, begin), bisect_right(t, end)
    if first < blocks.start or last > blocks.stop:
        log.warning(
            f'The focus plans pulses [{first}, {last}) past production [{blocks.start}, {blocks.stop}); '
            "reading only production's, so the edge lines' aperture is cut as in production"
        )
    start = max(blocks.start, blocks.start + (first - blocks.start) // size * size)
    stop = start + math.ceil((last - start) / size) * size
    if stop > blocks.stop - size:
        stop = blocks.stop
    log.info(f'RFI block alignment: reading pulses [{start}, {stop}) of production [{blocks.start}, {blocks.stop})')
    begin = (t[start - 1] + t[start]) / 2 if start > 0 else t[0] - 1
    end = (t[stop - 1] + t[stop]) / 2 if stop < len(t) else t[-1] + 1
    return begin, end


def plan_pulses(cfg: nisar_focus.Struct, skeleton_path: str, blocks: RfiBlocks | None) -> dict[str, tuple[int, int]]:
    """The pulse windows the focus will read for the runconfig's output grid, RFI-aligned when blocks are given.

    Args:
        cfg: The loaded runconfig's groups, naming the skeleton and carrying the AOI output grid.
        skeleton_path: Path of the skeleton.
        blocks: The production run's RFI blocks, or None.

    Returns:
        {tx: (start, stop)} for every transmit polarization in the file.
    """
    epoch, begin, end = needed_raw_span(cfg, focus_geometry(cfg))
    if blocks is not None:
        begin, end = align_rfi_blocks(blocks, begin, end)
    return pulse_windows(skeleton_path, epoch, begin, end)


def fill_pulses(skeleton_path: str, windows: dict[str, tuple[int, int]], src: h5py.File) -> None:
    """Write the pulse windows of every signal array from src into the skeleton.

    Every polarization is filled, not only the focused ones: for quad-pol data the focus reads the
    noise-only pulses of the other transmit's array.

    Args:
        skeleton_path: Path of the skeleton (signal arrays empty).
        windows: {tx: (start, stop)} from plan_pulses.
        src: Handle the signal arrays are read from (typically the remote L0B).
    """
    with h5py.File(skeleton_path, 'r+') as dst:
        signals = []

        def collect(path: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset) and is_signal(obj):
                signals.append(path)

        dst.visititems(collect)  # local, and the visit only reads object headers
        for path in signals:
            txs = [tx for tx in windows if f'/tx{tx}/' in f'/{path}/']
            if not txs:
                raise ValueError(f'No pulse window for signal array {path}; windows are for tx {sorted(windows)}')
            start, stop = windows[txs[0]]
            for i in range(start, stop, BLOCK_PULSES):
                j = min(i + BLOCK_PULSES, stop)
                dst[path][i:j] = src[path][i:j]
    log.info(f'Streamed pulses {windows} of {len(signals)} signal arrays into {skeleton_path}')


@contextlib.contextmanager
def read_filled_pulses(
    skeleton_path: str, windows: dict[str, tuple[int, int]], blocks: RfiBlocks | None
) -> Iterator[None]:
    """Run the focus on the filled skeleton: RFI-aligned span, and an error if it would read unfilled pulses.

    The focus picks its pulses by bisecting each channel's pulse times against nisar_focus.total_bounds;
    that is patched to widen the span (align_rfi_blocks) and to check it against the filled windows.
    Unfilled pulses would otherwise read silently as zeros. The focus must call it exactly once (as
    isce3 0.25 does); any other count means the patch did not govern the read, which is an error.

    Args:
        skeleton_path: Path of the filled skeleton, the focus input.
        windows: {tx: (start, stop)} that fill_pulses wrote.
        blocks: The production run's RFI blocks, or None.
    """
    orig_total = nisar_focus.total_bounds
    raw = open_rrsd(skeleton_path)
    epoch = nisar_focus.get_total_grid_bounds([skeleton_path])[0]
    calls = 0

    def total_bounds(blocks_bounds: list) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        begin, end = orig_total(blocks_bounds)
        if blocks is not None:
            begin, end = align_rfi_blocks(blocks, begin, end)
        for tx, (start, stop) in windows.items():
            times, _ = raw.getRadarGrid(raw.frequencies[0], tx=tx, epoch=epoch)
            if not (start <= bisect_left(times, begin) and bisect_right(times, end) <= stop):
                raise ValueError(f'The focus would read tx{tx} pulses outside the streamed [{start}, {stop})')
        return begin, end

    nisar_focus.total_bounds = total_bounds
    try:
        yield
    finally:
        nisar_focus.total_bounds = orig_total
    if calls != 1:
        raise RuntimeError(f'The focus called total_bounds {calls} times, not once; the pulse span went unchecked')
