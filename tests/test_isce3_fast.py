import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from isce3.atmosphere import ionosphere_filter
from isce3.math import offsets_polyfit
from nisar.workflows import crossmul, dense_offsets, helpers, ionosphere, rubbersheet
from osgeo import gdal

from hyp3_isce3 import isce3_fast, process


def test_polyfit_offsets_matches_isce3():
    # Tie points on a quadratic offset surface with noise and a few gross outliers for the fit to reject.
    rng = np.random.default_rng(0)
    n = 400
    lines, pixels = rng.uniform(0, 5000, n), rng.uniform(0, 3000, n)
    d_line = 0.3 + 1e-4 * lines - 5e-5 * pixels + rng.normal(0, 0.01, n)
    d_pixel = -0.1 + 2e-5 * lines + 3e-5 * pixels + rng.normal(0, 0.01, n)
    d_line[:10] += 5.0
    data = np.column_stack([np.arange(n), lines, pixels, d_line, d_pixel, rng.uniform(0.2, 1.0, n)])
    kwargs: dict[str, Any] = {
        'degree': 2,
        'crit_value': 3.0,
        'max_iterations': n,
        'prf': 1520.0,
        'abw': 1258.5,
        'rsr': 48e6,
        'rbw': 40e6,
    }

    expected = offsets_polyfit.polyfit_offsets(data.copy(), **kwargs)
    result = isce3_fast.polyfit_offsets(data.copy(), **kwargs)

    assert result['removed_indices'] == expected['removed_indices']
    assert len(result['removed_indices']) >= 10
    np.testing.assert_allclose(result['coefL'], expected['coefL'], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result['coefP'], expected['coefP'], rtol=1e-9, atol=1e-12)


def _image_with_gaps() -> np.ndarray:
    rng = np.random.default_rng(1)
    image = rng.normal(size=(40, 50)).astype(np.float32)
    image[rng.uniform(size=image.shape) < 0.2] = np.nan
    image[10:16, 20:28] = np.nan  # a hole wider than the filter windows
    return image


@pytest.mark.parametrize('size', [3, 5])
def test_nan_median_filter_matches_isce3(size):
    image = _image_with_gaps()
    expected = ionosphere_filter.nan_median_filter(image, size)
    np.testing.assert_allclose(isce3_fast.nan_median_filter(image, size), expected, rtol=1e-6, equal_nan=True)


def test_convolve_preserve_nan_matches_isce3():
    image = _image_with_gaps().astype(np.float64)
    rows, cols = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2), np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
    # an outer-product (Gaussian) kernel takes the two-pass path; a non-separable one falls back to isce3's function
    separable = np.outer(rows / rows.sum(), cols / cols.sum())
    lopsided = separable.copy()
    lopsided[0, 0] += 0.05
    for kernel in (separable, lopsided):
        expected = ionosphere_filter.convolve_preserve_nan(image, kernel)
        np.testing.assert_allclose(
            isce3_fast.convolve_preserve_nan(image, kernel), expected, rtol=1e-9, atol=1e-12, equal_nan=True
        )


def _rubbersheet_rasters(tmp_path, name: str) -> tuple[str, str, dict]:
    """A small culled offsets raster and a geo2rdr offsets raster with some invalid samples, as ENVI float64 files."""
    rng = np.random.default_rng(2)
    length, width = 700, 90
    for path, array in [
        (f'culled_{name}', rng.normal(size=(12, 9))),
        (f'geo_{name}', rng.normal(size=(length, width))),
    ]:
        array[2, 3] = -1e6
        raster = gdal.GetDriverByName('ENVI').Create(
            str(tmp_path / path), array.shape[1], array.shape[0], 1, gdal.GDT_Float64
        )
        raster.GetRasterBand(1).WriteArray(array)
        raster.FlushCache()
    # offsets grid rows and columns sit inside the RSLC grid, so its edges are extrapolated
    grid = {
        'off_az_pos': np.linspace(30, 650, 12),
        'off_rg_pos': np.linspace(5, 80, 9),
        'length': length,
        'width': width,
    }
    return str(tmp_path / f'culled_{name}'), str(tmp_path / f'geo_{name}'), grid


def _isce3_two_passes(tmp_path, culled: str, geo: str, grid: dict, name: str) -> bytes:
    resampled, summed = str(tmp_path / f'resampled_{name}'), tmp_path / f'expected_{name}'
    rubbersheet._resample_offsets_to_slc(culled, resampled, **grid)
    helpers.sum_gdal_rasters(geo, resampled, str(summed), invalid_value=-1e6)
    return summed.read_bytes()


def test_resample_and_sum_offsets_matches_isce3(tmp_path):
    culled, geo, grid = _rubbersheet_rasters(tmp_path, 'az')
    expected = _isce3_two_passes(tmp_path, culled, geo, grid, 'az')

    # several blocks, the last one short
    isce3_fast.resample_and_sum_offsets(
        culled, geo, str(tmp_path / 'fused'), **grid, invalid_value=-1e6, lines_per_block=256
    )

    assert (tmp_path / 'fused').read_bytes() == expected
    assert (np.frombuffer(expected, dtype=np.float64) == -1e6).any()


def test_overlap_rubbersheet_rasters_matches_isce3(tmp_path, monkeypatch):
    rasters = {name: _rubbersheet_rasters(tmp_path, name) for name in ('az', 'rg')}
    expected = {name: _isce3_two_passes(tmp_path, *rasters[name], name) for name in rasters}
    # put isce3's own functions back after the test
    monkeypatch.setattr(rubbersheet, '_resample_offsets_to_slc', rubbersheet._resample_offsets_to_slc)
    monkeypatch.setattr(rubbersheet, 'sum_gdal_rasters', rubbersheet.sum_gdal_rasters)
    isce3_fast._overlap_rubbersheet_rasters()

    # the calls rubbersheet makes, in its order: azimuth, then range
    for name, out in [('az', 'azimuth.off'), ('rg', 'range.off')]:
        culled, geo, grid = rasters[name]
        rubbersheet._resample_offsets_to_slc(culled, str(tmp_path / f'unwritten_{name}'), **grid)
        rubbersheet.sum_gdal_rasters(geo, str(tmp_path / f'unwritten_{name}'), str(tmp_path / out), invalid_value=-1e6)

    assert (tmp_path / 'azimuth.off').read_bytes() == expected['az']
    assert (tmp_path / 'range.off').read_bytes() == expected['rg']
    assert not (tmp_path / 'unwritten_az').exists()
    # range before azimuth is refused
    culled, geo, grid = rasters['rg']
    rubbersheet._resample_offsets_to_slc(culled, str(tmp_path / 'unwritten_rg'), **grid)
    with pytest.raises(RuntimeError, match='azimuth-then-range'):
        rubbersheet.sum_gdal_rasters(
            geo, str(tmp_path / 'unwritten_rg'), str(tmp_path / 'range.off'), invalid_value=-1e6
        )


def test_reuse_slc_copies(tmp_path, monkeypatch):
    copies = []

    def fake_copy(infile, freq, pol, lines_per_block, outfile, file_type='ENVI'):
        copies.append(outfile)
        Path(outfile).write_bytes(f'{infile} {freq} {pol}'.encode())
        Path(outfile).with_suffix('.hdr').write_text('ENVI')

    monkeypatch.setattr(crossmul, 'copy_raster', fake_copy)
    monkeypatch.setattr(dense_offsets, 'copy_raster', fake_copy)
    monkeypatch.setattr(isce3_fast, '_SLC_COPIES', {})
    isce3_fast._reuse_slc_copies()
    first, second, other = (str(tmp_path / name) for name in ('first.slc', 'second.slc', 'other.slc'))

    dense_offsets.copy_raster('ref.h5', 'A', 'HH', 1000, first, file_type='ENVI')
    crossmul.copy_raster('ref.h5', 'A', 'HH', 1024, second, file_type='ENVI')
    # asked for again: the link is replaced, not written through
    crossmul.copy_raster('ref.h5', 'A', 'HH', 1024, second, file_type='ENVI')
    crossmul.copy_raster('ref.h5', 'A', 'HV', 1024, other, file_type='ENVI')

    # one real copy per RSLC dataset; the repeat is the same file under a second name, header included
    assert copies == [first, other]
    assert (tmp_path / 'second.slc').samefile(first)
    assert (tmp_path / 'second.hdr').samefile(tmp_path / 'first.hdr')
    assert (tmp_path / 'first.slc').read_bytes() == b'ref.h5 A HH'
    assert (tmp_path / 'other.slc').read_bytes() == b'ref.h5 A HV'

    # the first copy is gone: copy again as isce3 does
    (tmp_path / 'first.slc').unlink()
    (tmp_path / 'second.slc').unlink()
    crossmul.copy_raster('ref.h5', 'A', 'HH', 1024, second, file_type='ENVI')
    assert copies == [first, other, second]


class _FakeProcess:
    exitcode = 0

    def __init__(self, target, args):
        self.args, self.started, self.joined = args, False, False

    def start(self):
        self.started = True

    def join(self):
        self.joined = True


def _iono_cfg(method: str, side_band_pols: list | None) -> dict:
    return {
        'input_file_group': {'reference_rslc_file': 'ref.h5', 'secondary_rslc_file': 'sec.h5'},
        'processing': {
            'input_subset': {'list_of_frequencies': {'A': ['HH']}},
            'ionosphere_phase_correction': {
                'enabled': True,
                'spectral_diversity': method,
                'list_of_frequencies': {'A': ['HH'], 'B': side_band_pols},
            },
        },
    }


def test_start_side_band(monkeypatch):
    isce3_workflow = ionosphere.run_insar_workflow
    monkeypatch.setattr(ionosphere, 'run_insar_workflow', isce3_workflow)
    monkeypatch.setattr(isce3_fast.SPAWN, 'Process', _FakeProcess)

    # nothing to overlap: a split-spectrum method, or no polarization beyond the main run's
    assert isce3_fast.start_side_band(_iono_cfg('split_main_band', ['HH']), 'insar.yaml', 'RUNW.h5') is None
    assert isce3_fast.start_side_band(_iono_cfg('main_diff_ms_band', None), 'insar.yaml', 'RUNW.h5') is None
    assert ionosphere.run_insar_workflow is isce3_workflow

    # Any: the stand-in process class records what the real one would be asked to do
    child: Any = isce3_fast.start_side_band(_iono_cfg('main_diff_ms_band', ['HH']), 'insar.yaml', 'RUNW.h5')
    assert child.started and not child.joined
    assert child.args == ('insar.yaml', {'reference_rslc_file': 'ref.h5', 'secondary_rslc_file': 'sec.h5'}, 'RUNW.h5')
    # the ionosphere step's request for the side-band interferogram collects the child, once
    ionosphere.run_insar_workflow({}, {}, {}, unwrapping_flag=False)
    assert child.joined
    assert ionosphere.run_insar_workflow is isce3_workflow


def test_only_steps():
    steps = isce3_fast._only_steps('rdr2geo', 'geo2rdr')
    assert {step for step, on in steps.items() if on} == {'rdr2geo', 'geo2rdr'}
    # every step insar.run looks up has to be present, on or off
    assert {'bandpass_insar', 'prepare_insar_hdf5', 'unwrap', 'ionosphere', 'baseline'} <= set(steps)


@pytest.mark.parametrize('flag, fast', [([], False), (['--fast'], True)])
def test_main_passes_fast_flag(monkeypatch, flag, fast):
    calls = []
    monkeypatch.setattr(process, 'process_isce3', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['gunw', 'granule_a', 'granule_b', *flag])
    process.main()
    assert calls == [{'subset': None, 'fast': fast}]


def test_use_whirlwind(tmp_path):
    runconfig_path = tmp_path / 'insar.yaml'
    processing = {'phase_unwrap': {'algorithm': 'snaphu', 'range_looks': 13}, 'crossmul': {'range_looks': 5}}
    runconfig_path.write_text(yaml.safe_dump({'runconfig': {'groups': {'processing': processing}}}))
    cfg = {'processing': {'phase_unwrap': {'algorithm': 'snaphu', 'range_looks': 13}}}

    isce3_fast._use_whirlwind(cfg, str(runconfig_path))

    # both the loaded runconfig and the yaml the children reload ask for whirlwind; nothing else moves
    assert cfg['processing']['phase_unwrap'] == {'algorithm': 'whirlwind', 'range_looks': 13}
    written = yaml.safe_load(runconfig_path.read_text())['runconfig']['groups']['processing']
    assert written == {'phase_unwrap': {'algorithm': 'whirlwind', 'range_looks': 13}, 'crossmul': {'range_looks': 5}}
