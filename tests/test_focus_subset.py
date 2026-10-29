from bisect import bisect_left, bisect_right
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, cast

import h5py
import isce3
import numpy as np
import pytest
from isce3.noise.noise_power_est_func import cpi_slice_gen
from nisar.workflows import focus as nisar_focus

from hyp3_isce3.focus import subset


N_PULSES = 100
SIGNAL_DTYPE = np.dtype([('r', '<u2'), ('i', '<u2')])


@pytest.fixture
def tiny_l0b(tmp_path):
    """An L0B-shaped file: pulse-indexed datasets, a chunked signal array, and whole-file metadata."""
    path = tmp_path / 'l0b.h5'
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        f.attrs['Conventions'] = 'CF-1.7'
        ident = f.create_group('science/LSAR/identification')
        ident.create_dataset('zeroDopplerStartTime', data=np.bytes_('2025-11-04T00:16:40.000000000'))
        ident.create_dataset('zeroDopplerEndTime', data=np.bytes_('2025-11-04T00:16:40.066000000'))
        tx = f.create_group('science/LSAR/RRSD/swaths/frequencyA/txH')
        tx.attrs['description'] = 'transmit H'
        utc = tx.create_dataset('UTCtime', data=1000.0 + np.arange(N_PULSES) / 1500.0)
        utc.attrs['units'] = 'seconds since 2025-11-04T00:00:00'
        tx.create_dataset('calType', data=np.zeros(N_PULSES, dtype='u1'))
        rx = tx.create_group('rxH')
        signal = np.zeros((N_PULSES, 16), dtype=SIGNAL_DTYPE)
        signal['r'], signal['i'] = rng.integers(0, 65535, (N_PULSES, 16)), rng.integers(0, 65535, (N_PULSES, 16))
        rx.create_dataset('HH', data=signal, chunks=(8, 8), compression='gzip', compression_opts=1)
        rx.create_dataset('caltone', data=rng.standard_normal((N_PULSES, 12)).astype('c8'))
        rx.create_dataset('BFPQLUT', data=np.arange(16, dtype='f4'))
        f.create_dataset('science/LSAR/RRSD/highRateTelemetry/txH/rxH/QFSP0/X', data=np.arange(N_PULSES, dtype='f4'))
        f.create_dataset('science/LSAR/RRSD/lowRateTelemetry/attitude/time', data=np.arange(10, dtype='f8'))
    return path


@pytest.fixture
def skeleton(tiny_l0b, tmp_path):
    with h5py.File(tiny_l0b, 'r') as src:
        return subset.write_skeleton(src, tmp_path / 'skeleton.h5')


def test_write_skeleton_leaves_only_the_signal_empty(tiny_l0b, skeleton):
    with h5py.File(tiny_l0b, 'r') as src, h5py.File(skeleton, 'r') as dst:
        swaths = 'science/LSAR/RRSD/swaths/frequencyA/txH'
        signal = dst[f'{swaths}/rxH/HH']
        assert (signal.shape, signal.chunks, signal.compression) == ((N_PULSES, 16), (8, 8), 'gzip')
        assert not signal[()]['r'].any()
        for path in (
            f'{swaths}/UTCtime',
            f'{swaths}/rxH/caltone',
            'science/LSAR/RRSD/highRateTelemetry/txH/rxH/QFSP0/X',
            'science/LSAR/RRSD/lowRateTelemetry/attitude/time',
        ):
            assert np.array_equal(src[path][()], dst[path][()])
        assert dst[f'{swaths}/UTCtime'].attrs['units'] == src[f'{swaths}/UTCtime'].attrs['units']
        assert dst[swaths].attrs['description'] == 'transmit H'
        assert dst.attrs['Conventions'] == 'CF-1.7'


class FakeRemoteFile:
    """Stand-in for an earthaccess remote file: opens a local path and records closing."""

    def __init__(self, path):
        self.path = path
        self.handle = path.open('rb')  # noqa: SIM115
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def close(self):
        self.handle.close()
        self.closed = True


# The fields of a CMR granule that open_remote_l0b reads.
GRANULE = cast(subset.earthaccess.DataGranule, {'umm': {'GranuleUR': 'l0b'}})


def test_open_remote_l0b_closes_every_granule_file(tiny_l0b, tmp_path, monkeypatch):
    qa = tmp_path / 'l0b_QA_STATS.h5'
    qa.write_bytes(b'')
    files = [FakeRemoteFile(tiny_l0b), FakeRemoteFile(qa)]
    monkeypatch.setattr(subset.earthaccess, 'open', lambda granules: files)
    with subset.open_remote_l0b(GRANULE) as remote:
        assert 'science/LSAR/identification' in remote
        assert not any(f.closed for f in files)
    assert all(f.closed for f in files)


def test_open_remote_l0b_without_the_h5_closes_and_raises(tmp_path, monkeypatch):
    qa = tmp_path / 'l0b_QA_STATS.h5'
    qa.write_bytes(b'')
    files = [FakeRemoteFile(qa)]
    monkeypatch.setattr(subset.earthaccess, 'open', lambda granules: files)
    with pytest.raises(RuntimeError, match='No l0b.h5'):
        with subset.open_remote_l0b(GRANULE):
            pass
    assert files[0].closed


def test_fill_pulses_writes_only_the_window_of_the_signal(tiny_l0b, skeleton):
    with h5py.File(tiny_l0b, 'r') as remote:
        subset.fill_pulses(str(skeleton), {'H': (10, 40)}, remote)
    with h5py.File(tiny_l0b, 'r') as src, h5py.File(skeleton, 'r') as dst:
        path = 'science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH'
        assert np.array_equal(dst[path][10:40], src[path][10:40])
        assert not dst[path][:10]['r'].any() and not dst[path][40:]['r'].any()


def test_fill_pulses_without_a_window_for_the_transmit_raises(tiny_l0b, skeleton):
    with h5py.File(tiny_l0b, 'r') as remote, pytest.raises(ValueError, match='No pulse window'):
        subset.fill_pulses(str(skeleton), {'V': (10, 40)}, remote)


def test_snap_to_full_frame_rounds_outward_onto_block_multiples():
    epoch = isce3.core.DateTime('2025-11-04T00:00:00')
    grid = isce3.product.RadarGridParameters(100.0, 0.24, 1520.0, 900000.0, 3.0, 'left', 3040, 1000, epoch)
    t_lo, t_hi = 100.0 + 100 / 1520, 100.0 + 2000 / 1520  # lines 100 and 2000
    r_lo, r_hi = 900000.0 + 600 * 3.0, 900000.0 + 1500 * 3.0  # samples 600 and 1500 (past the grid's 1000)
    out = subset.snap_to_full_frame(t_lo, t_hi, r_lo, r_hi, grid, az_block=48, rg_block=520)
    start = datetime(2025, 11, 4, tzinfo=UTC)
    # lines 96 and 2016, samples 520 and 1000 (clipped to the grid)
    assert abs((out['start_time'] - start).total_seconds() - (100.0 + 96 / 1520)) < 1e-6
    assert abs((out['end_time'] - start).total_seconds() - (100.0 + 2016 / 1520)) < 1e-6
    assert out['start_range'] == 900000.0 + 520 * 3.0
    assert out['end_range'] == 900000.0 + 1000 * 3.0


def test_snap_to_full_frame_with_exact_block_lands_on_whole_microseconds():
    grid = isce3.product.RadarGridParameters(
        7.0, 0.24, 1520.0, 900000.0, 3.0, 'left', 100000, 1000, isce3.core.DateTime('2025-11-04T00:00:00')
    )
    block = 48 * 19
    for line in (100, 5000, 77777):
        out = subset.snap_to_full_frame(
            7.0 + line / 1520, 7.0 + (line + 1) / 1520, 900000.0, 900300.0, grid, block, 520
        )
        seconds = (out['start_time'] - datetime(2025, 11, 4, tzinfo=UTC)).total_seconds()
        start_line = (seconds - 7.0) * 1520
        assert abs(start_line - round(start_line)) < 1e-9  # exactly on the full-frame line lattice
        assert round(start_line) % block == 0


def test_snap_to_full_frame_outside_the_frame_raises():
    grid = isce3.product.RadarGridParameters(
        100.0, 0.24, 1520.0, 900000.0, 3.0, 'left', 3040, 1000, isce3.core.DateTime('2025-11-04T00:00:00')
    )
    with pytest.raises(ValueError, match='overlap'):
        subset.snap_to_full_frame(200.0, 201.0, 900000.0, 900300.0, grid, 48, 520)


def blocks(start, stop, n=5000):
    """Production RFI blocks of 512 pulses over [start, stop) of 1 kHz pulses."""
    return subset.RfiBlocks('H', np.arange(n) / 1000.0, start, stop, 512)


def aligned(rfi_blocks, first, last):
    """The pulse window align_rfi_blocks gives for a plan needing pulses first..last."""
    t = rfi_blocks.times
    begin, end = subset.align_rfi_blocks(rfi_blocks, t[first], t[last])
    return bisect_left(t, begin), bisect_right(t, end)


def test_align_rfi_blocks_tiles_the_production_blocks():
    # Start one block after production's first pulse, then three whole blocks.
    assert aligned(blocks(100, 4700), 1000, 2000) == (612, 2148)


def test_align_rfi_blocks_clips_to_the_production_window(caplog):
    assert aligned(blocks(100, 4700), 50, 4800) == (100, 4700)
    assert 'past production' in caplog.text


def test_align_rfi_blocks_inside_production_does_not_warn(caplog):
    aligned(blocks(100, 4700), 1000, 2000)
    assert 'past production' not in caplog.text


def test_align_rfi_blocks_takes_the_production_end_inside_its_slid_last_block():
    # Whole blocks would stop at 4196, inside production's slid last block [4188, 4700).
    assert aligned(blocks(100, 4700), 3700, 4100) == (3684, 4700)


def test_aligned_window_tiles_into_production_blocks():
    # Every block isce3 cuts from the aligned window is one of the blocks it cuts from production's.
    rfi_blocks = blocks(100, 4700)
    production = {(s.start + 100, s.stop + 100) for s in cpi_slice_gen(4600, 512)}
    for first, last in ((1000, 2000), (3700, 4100), (150, 600), (4000, 4650)):
        start, stop = aligned(rfi_blocks, first, last)
        ours = {(s.start + start, s.stop + start) for s in cpi_slice_gen(stop - start, 512)}
        assert ours <= production


# These tests never reach the geometry: the RFI gate returns first, or needed_raw_span is stubbed.
NO_GEOMETRY = cast(subset.FocusGeometry, None)


def rfi_cfg(algorithm='tone-rank', mitigation=True):
    rfi = SimpleNamespace(
        detection_enabled=True,
        mitigation_enabled=mitigation,
        mitigation_algorithm=algorithm,
        tone_rank=SimpleNamespace(block_dims=[512, 1024]),
    )
    return SimpleNamespace(processing=SimpleNamespace(radio_frequency_interference=rfi))


class QuadPolRaw:
    """Stand-in for an L0B reader with both transmit polarizations."""

    frequencies: ClassVar[list[str]] = ['A']
    polarizations: ClassVar[dict[str, list[str]]] = {'A': ['HH', 'HV', 'VH', 'VV']}

    def getRadarGrid(self, frequency, tx, epoch):
        return np.arange(5000) / 1000.0, None


def test_production_rfi_blocks_only_for_tone_rank_mitigation():
    assert subset.production_rfi_blocks(rfi_cfg('ST-EVD'), NO_GEOMETRY, 'skeleton.h5', None) is None
    assert subset.production_rfi_blocks(rfi_cfg(mitigation=False), NO_GEOMETRY, 'skeleton.h5', None) is None


def test_production_rfi_blocks_needs_one_transmit(monkeypatch):
    monkeypatch.setattr(subset, 'open_rrsd', lambda path: QuadPolRaw())
    monkeypatch.setattr(subset, 'needed_raw_span', lambda cfg, geometry: (None, 1.0, 2.0))
    with pytest.raises(NotImplementedError, match='one transmit'):
        subset.production_rfi_blocks(rfi_cfg(), NO_GEOMETRY, 'skeleton.h5', None)
    # Restricting the focus to one transmit is fine on a quad-pol file.
    rfi_blocks = subset.production_rfi_blocks(rfi_cfg(), NO_GEOMETRY, 'skeleton.h5', ['HH', 'HV'])
    assert rfi_blocks is not None
    assert (rfi_blocks.tx, rfi_blocks.start, rfi_blocks.stop, rfi_blocks.size) == ('H', 1000, 2001, 512)


def test_read_filled_pulses_refuses_unfilled_pulses_and_restores(monkeypatch):
    monkeypatch.setattr(subset, 'open_rrsd', lambda path: QuadPolRaw())
    monkeypatch.setattr(nisar_focus, 'get_total_grid_bounds', lambda paths: (None,))
    original = nisar_focus.total_bounds
    t = np.arange(5000) / 1000.0
    with subset.read_filled_pulses('skeleton.h5', {'H': (1000, 2001)}, None):
        assert nisar_focus.total_bounds([(None, (t[1500], t[2000]))]) == (t[1500], t[2000])
    assert nisar_focus.total_bounds is original
    with pytest.raises(ValueError, match='outside the streamed'):
        with subset.read_filled_pulses('skeleton.h5', {'H': (1000, 2001)}, None):
            nisar_focus.total_bounds([(None, (t[900], t[2000]))])
    assert nisar_focus.total_bounds is original


def test_read_filled_pulses_needs_exactly_one_total_bounds_call(monkeypatch):
    monkeypatch.setattr(subset, 'open_rrsd', lambda path: QuadPolRaw())
    monkeypatch.setattr(nisar_focus, 'get_total_grid_bounds', lambda paths: (None,))
    t = np.arange(5000) / 1000.0
    with pytest.raises(RuntimeError, match='0 times'):
        with subset.read_filled_pulses('skeleton.h5', {'H': (1000, 2001)}, None):
            pass
    with pytest.raises(RuntimeError, match='2 times'):
        with subset.read_filled_pulses('skeleton.h5', {'H': (1000, 2001)}, None):
            nisar_focus.total_bounds([(None, (t[1500], t[2000]))])
            nisar_focus.total_bounds([(None, (t[1500], t[2000]))])
