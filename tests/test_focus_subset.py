from datetime import UTC, datetime

import h5py
import isce3
import numpy as np
import pytest
from nisar.antenna import pattern
from nisar.products.readers.Raw.Raw import RawBase
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


def test_write_skeleton_leaves_only_the_signal_empty(tiny_l0b, tmp_path):
    with h5py.File(tiny_l0b, 'r') as src:
        skeleton = subset.write_skeleton(src, tmp_path / 'skeleton.h5')
    with h5py.File(tiny_l0b, 'r') as src, h5py.File(skeleton, 'r') as dst:
        signal = dst['science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH']
        assert (signal.shape, signal.chunks, signal.compression) == ((N_PULSES, 16), (8, 8), 'gzip')
        assert not signal[()]['r'].any()
        for path in (
            'science/LSAR/RRSD/swaths/frequencyA/txH/UTCtime',
            'science/LSAR/RRSD/swaths/frequencyA/txH/rxH/caltone',
            'science/LSAR/RRSD/highRateTelemetry/txH/rxH/QFSP0/X',
            'science/LSAR/RRSD/lowRateTelemetry/attitude/time',
        ):
            assert np.array_equal(src[path][()], dst[path][()])
        assert (
            dst['science/LSAR/RRSD/swaths/frequencyA/txH/UTCtime'].attrs['units']
            == src['science/LSAR/RRSD/swaths/frequencyA/txH/UTCtime'].attrs['units']
        )
        assert dst['science/LSAR/RRSD/swaths/frequencyA/txH'].attrs['description'] == 'transmit H'
        assert dst.attrs['Conventions'] == 'CF-1.7'


def test_crop_l0b_slices_pulse_datasets_and_streams_the_signal(tiny_l0b, tmp_path):
    with h5py.File(tiny_l0b, 'r') as src:
        skeleton = subset.write_skeleton(src, tmp_path / 'skeleton.h5')
    windows = {'H': (10, 40)}
    with h5py.File(tiny_l0b, 'r') as remote:
        crop = subset.crop_l0b(skeleton, tmp_path / 'crop.h5', windows, signal_src=remote)

    with h5py.File(tiny_l0b, 'r') as src, h5py.File(crop, 'r') as dst:
        swaths = 'science/LSAR/RRSD/swaths/frequencyA/txH'
        # pulse-indexed datasets are sliced, the signal from the remote handle
        assert np.array_equal(dst[f'{swaths}/UTCtime'][()], src[f'{swaths}/UTCtime'][10:40])
        assert np.array_equal(dst[f'{swaths}/rxH/HH'][()], src[f'{swaths}/rxH/HH'][10:40])
        assert np.array_equal(dst[f'{swaths}/rxH/caltone'][()], src[f'{swaths}/rxH/caltone'][10:40])
        telemetry = 'science/LSAR/RRSD/highRateTelemetry/txH/rxH/QFSP0/X'
        assert np.array_equal(dst[telemetry][()], src[telemetry][10:40])
        # layout is kept, chunks clamped to the cropped size
        assert dst[f'{swaths}/rxH/HH'].chunks == (8, 8)
        assert dst[f'{swaths}/rxH/HH'].compression == 'gzip'
        # everything without a pulse axis is untouched
        assert np.array_equal(dst[f'{swaths}/rxH/BFPQLUT'][()], src[f'{swaths}/rxH/BFPQLUT'][()])
        lowrate = 'science/LSAR/RRSD/lowRateTelemetry/attitude/time'
        assert np.array_equal(dst[lowrate][()], src[lowrate][()])
        # identification times follow the kept pulses (t = 1000 + i/1500 s after midnight, isce3 rounds to ns)
        ident = dst['science/LSAR/identification']
        assert ident['zeroDopplerStartTime'][()].decode() == '2025-11-04T00:16:40.006666667'
        assert ident['zeroDopplerEndTime'][()].decode() == '2025-11-04T00:16:40.026000000'


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


def test_microsecond_exact_lines():
    # 1520 = 16 * 5 * 19; a microsecond lattice absorbs the 16 and the 5
    assert subset.microsecond_exact_lines(1520) == 19
    assert subset.microsecond_exact_lines(1000) == 1
    with pytest.raises(ValueError, match='whole number'):
        subset.microsecond_exact_lines(1520.5)


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


def test_full_file_quantities_patches_and_restores(tmp_path):
    originals = (
        RawBase.getRadarGrid,
        RawBase.getNominalPRF,
        nisar_focus.get_total_grid_bounds,
        pattern.compute_all_rx_channel_imbalances_from_l0b,
    )
    with subset.full_file_quantities(str(tmp_path / 'skeleton.h5')):
        patched = (
            RawBase.getRadarGrid,
            RawBase.getNominalPRF,
            nisar_focus.get_total_grid_bounds,
            pattern.compute_all_rx_channel_imbalances_from_l0b,
        )
        assert all(p is not o for p, o in zip(patched, originals))
    restored = (
        RawBase.getRadarGrid,
        RawBase.getNominalPRF,
        nisar_focus.get_total_grid_bounds,
        pattern.compute_all_rx_channel_imbalances_from_l0b,
    )
    assert all(r is o for r, o in zip(restored, originals))


def test_full_file_quantities_none_is_a_no_op():
    original = RawBase.getRadarGrid
    with subset.full_file_quantities(None):
        assert RawBase.getRadarGrid is original
