"""Faster isce3 InSAR run for ``++process gunw --fast``.

The same steps and runconfig as nisar.workflows.insar.run, with idle cores put to work. Its products are NOT
bit-identical to the standard run's. Measured on a full frame (T120 F023, isce3 0.25.22, 2026-11-18): CPU dense offsets
land in a second reproducible state when the product writers have not run earlier in the same process (about 60 of
1.07M windows move by 1/32 px), and the rubbersheet fit turns that into a smooth surface of about +-2 mm along-track
and +-1.5 mrad across the frame. The cause inside isce3 is not identified.

Everything that departs from stock isce3 lives in this file, in three groups:

  faster functions  Three slow pure-numpy isce3 functions are replaced with equivalents: the rubbersheet polynomial
                    fit (one einsum written as a matrix product) and the ionosphere NaN-median and NaN-aware
                    Gaussian filters. An SLC that dense offsets or crossmul has already copied out of its HDF5
                    is hard-linked for the next step that asks for it instead of being copied again.
  parallel steps    CPU dense offsets run as one process per azimuth strip of the offset grid. The crossmul that
                    unwrap reruns at its own looks starts in a child process next to the first crossmul.
                    Rubbersheet's azimuth and range offsets for fine resample are made side by side, each in one
                    pass over the full RSLC grid instead of two (the files come out byte-identical to isce3's).
  overlapped steps  The product writers run next to coarse resample and dense offsets. The wrapped-interferogram
                    geocoding, troposphere and solid earth tides run while the single-threaded unwrap is going.
                    start_side_band would also run the ionosphere's frequency B interferogram during unwrap; it is
                    here but fast_run does not call it, as it has not been run on a real pair.

The unwrapper is also switched from the runconfig's snaphu to whirlwind, isce3's other minimum-cost-flow
unwrapper, so the unwrapped phase and connected components come from a different algorithm than a standard run's.

The first two groups replace functions inside isce3's modules for this process (see :func:`fast_run`); isce3 has no
supported way to do either. Child processes rebuild the runconfig from the yaml, since the loaded dictionary holds
isce3 objects that do not pickle.
"""

import argparse
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import isce3
import numpy as np
import yaml
from isce3.atmosphere import ionosphere_filter
from isce3.io.gdal.gdal_raster import GDALRaster
from isce3.math import offsets_polyfit
from nisar.products.readers import SLC
from nisar.workflows import (
    crossmul,
    dense_offsets,
    geocode_insar,
    h5_prep,
    insar,
    ionosphere,
    prepare_insar_hdf5,
    rubbersheet,
)
from nisar.workflows.geocode_insar import InputProduct
from nisar.workflows.helpers import get_cfg_freq_pols
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence
from osgeo import gdal
from scipy.ndimage import convolve1d


log = logging.getLogger(__name__)

# Spawned, not forked: the parent has already run OpenMP regions, and a forked child that starts its own can hang.
SPAWN = multiprocessing.get_context('spawn')
# rasters CPU ampcor writes, and their band counts (band-interleaved-by-pixel float32)
OFFSET_RASTERS = {'dense_offsets': 2, 'gross_offset': 2, 'snr': 1, 'covariance': 3, 'correlation_peak': 1}
# flat copies of SLCs made so far in this run: (RSLC path, frequency, polarization, format) -> copy
_SLC_COPIES: dict[tuple, str] = {}


# === faster functions ================================================================================================
_isce3_convolve_preserve_nan = ionosphere_filter.convolve_preserve_nan


def polyfit_offsets(
    data: np.ndarray,
    degree: int = 2,
    crit_value: float = 0.1,
    max_iterations: int = 50,
    minL: float | None = None,
    maxL: float | None = None,
    minP: float | None = None,
    maxP: float | None = None,
    prf: float | None = None,
    abw: float | None = None,
    rsr: float | None = None,
    rbw: float | None = None,
) -> dict:
    """isce3.math.offsets_polyfit.polyfit_offsets, line for line, except the leverage term (marked)."""
    op = offsets_polyfit
    # prior standard deviations for the w-test
    sigma_l = 0.15 / (prf / abw if prf is not None and abw is not None else 1.1)
    sigma_p = 0.10 / (rsr / rbw if rsr is not None and rbw is not None else 1.1)
    maxL = data[:, 1].max() if maxL is None else maxL
    minL = data[:, 1].min() if minL is None else minL
    maxP = data[:, 2].max() if maxP is None else maxP
    minP = data[:, 2].min() if minP is None else minP
    if minL >= maxL or minP >= maxP:
        raise ValueError('polyfit_offsets needs minL < maxL and minP < maxP')

    n_unknowns = op.ncoeffs(degree)
    eps = np.sqrt(np.finfo(float).eps)
    removed_indices: list[int] = []
    design = op.build_design_matrix(data[:, 1], data[:, 2], degree, minL, maxL, minP, maxP)
    iteration = 0
    while True:
        y_l, y_p = data[:, 3:4], data[:, 4:5]
        if data.shape[0] <= n_unknowns:
            raise ValueError('No sufficient points for the rubbersheet polyfitting')

        # weighted normal equations
        w = np.clip(data[:, 5], eps, 1.0)
        qy_diag = 1.0 / (w * w)
        weights = w[:, None]
        design_w = design * weights
        normal = design_w.T @ design_w
        rhs_l, rhs_p = design_w.T @ (y_l * weights), design_w.T @ (y_p * weights)
        try:
            x_l, chol = op.cholesky_solve(normal, rhs_l)
            x_p, _ = op.cholesky_solve(normal, rhs_p)
        except np.linalg.LinAlgError:
            eye = np.eye(normal.shape[0])
            x_l, chol = op.cholesky_solve(normal + eps * eye, rhs_l)
            x_p, _ = op.cholesky_solve(normal + eps * eye, rhs_p)
        qx_hat = op.invert_from_cholesky(chol)

        # residuals
        e_l, e_p = y_l - design @ x_l, y_p - design @ x_p
        # changed: isce3 has np.einsum("ij,jk,ik->i", A, Qx_hat, A), the same sum done here as one matrix product
        qyhat_diag = ((design @ qx_hat) * design).sum(axis=1)
        s = np.sqrt(np.clip(qy_diag - qyhat_diag, eps, None))

        # w-test
        w_l, w_p = (e_l[:, 0] / (s * sigma_l)), (e_p[:, 0] / (s * sigma_p))
        max_any = max(np.abs(w_l).max(), np.abs(w_p).max())
        if (max_any <= crit_value) or (iteration >= max_iterations):
            return {
                'coefL': x_l[:, 0],
                'coefP': x_p[:, 0],
                'inliers': data,
                'removed_indices': removed_indices,
                'degree': degree,
                'design_nunk': n_unknowns,
            }

        # remove the worst outlier and fit again
        worst = int(np.argmax(w_l * w_l + w_p * w_p))
        removed_indices.append(int(data[worst, 0]))
        data = np.delete(data, worst, axis=0)
        design = np.delete(design, worst, axis=0)
        iteration += 1


def nan_median_filter(image: np.ndarray, size: int) -> np.ndarray:
    """NaN-ignoring median in a size x size window, with NaN outside the image, as isce3's nan_median_filter."""
    half = size // 2
    padded = np.pad(image.astype(np.float64), half, constant_values=np.nan)
    rows, cols = image.shape
    # one shifted copy of the image per window position
    stack = np.stack([padded[i : i + rows, j : j + cols] for i in range(size) for j in range(size)])
    with np.errstate(all='ignore'):
        return np.nanmedian(stack, axis=0).astype(image.dtype)


def convolve_preserve_nan(
    image: np.ndarray, kernel: np.ndarray, mode: str = 'constant', cval: float = 0.0
) -> np.ndarray:
    """isce3's convolve_preserve_nan, with an outer-product kernel applied as two 1-D passes."""
    # the 1-D factors of a normalized outer-product kernel are its row and column sums
    kernel_rows, kernel_cols = kernel.sum(axis=1), kernel.sum(axis=0)
    separable = np.allclose(kernel, np.outer(kernel_rows, kernel_cols), rtol=0, atol=1e-12 * np.abs(kernel).max())
    # anything else goes to isce3's own function: other kernels, even sizes, other edge handling
    if not separable or mode != 'constant' or cval != 0.0 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        return _isce3_convolve_preserve_nan(image, kernel, mode=mode, cval=cval)

    def smooth(array: np.ndarray) -> np.ndarray:
        rows_done = convolve1d(array, kernel_rows, axis=0, mode='constant', cval=0.0)
        return convolve1d(rows_done, kernel_cols, axis=1, mode='constant', cval=0.0)

    valid = ~np.isnan(image)
    weight = smooth(valid.astype(float))
    smoothed = smooth(np.where(valid, image, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed = np.divide(smoothed, weight, out=np.zeros_like(smoothed), where=weight > 1e-8)
    # no valid pixel under the kernel
    smoothed[weight <= 1e-8] = np.nan
    return smoothed


def _reuse_slc_copies() -> None:
    """Replace copy_raster in dense offsets and crossmul so an SLC is copied out of its HDF5 only once.

    Dense offsets, the first crossmul and the unwrap-looks crossmul each ask for the same flat copy of the
    reference SLC (24 GB on a full frame, decompressed on one core). The stand-in hard-links the copy made
    earlier in the run when the RSLC, frequency, polarization and format match, and copies as isce3 does
    otherwise. fast_run keeps scratch until its last step, so the earlier copy is still there.
    """
    isce3_copy = crossmul.copy_raster

    def copy_raster(
        infile: str, freq: str, pol: str, lines_per_block: int, outfile: str, file_type: str = 'ENVI'
    ) -> None:
        key = (os.path.realpath(infile), freq, pol, file_type)
        made = _SLC_COPIES.get(key)
        # the raster and its ENVI header
        targets = [Path(outfile), Path(outfile).with_suffix('.hdr')]
        # never write through a link left by an earlier call: that would overwrite the copy it points to
        for target in targets:
            target.unlink(missing_ok=True)
        if made is not None:
            try:
                for source, target in zip([Path(made), Path(made).with_suffix('.hdr')], targets):
                    os.link(source, target)
                log.info(f'{outfile} linked to the copy at {made}')
                return
            except OSError:
                # the earlier copy is gone, or sits on another filesystem: copy as isce3 does
                for target in targets:
                    target.unlink(missing_ok=True)
        isce3_copy(infile, freq, pol, lines_per_block, outfile, file_type=file_type)
        _SLC_COPIES[key] = outfile

    crossmul.copy_raster = copy_raster
    dense_offsets.copy_raster = copy_raster


# === parallel steps ==================================================================================================
def _load_cfg(yaml_path: str) -> dict:
    """Load the InSAR runconfig the way process.py does."""
    return InsarRunConfig(argparse.Namespace(run_config_path=str(yaml_path), log_file=False)).cfg


def _join(process: multiprocessing.process.BaseProcess, name: str) -> None:
    """Wait for a child and fail loudly if it failed."""
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f'{name} child process exited with code {process.exitcode}')


def _new_ampcor(ref_path: str, ref_shape: tuple, sec_path: str, sec_shape: tuple) -> isce3.matchtemplate.PyCPUAmpcor:
    """A CPU ampcor object pointed at the memory-mappable reference and secondary SLCs."""
    ampcor = isce3.matchtemplate.PyCPUAmpcor()
    ampcor.useMmap = 1
    ampcor.referenceImageName, ampcor.secondaryImageName = ref_path, sec_path
    ampcor.referenceImageHeight, ampcor.referenceImageWidth = ref_shape
    ampcor.secondaryImageHeight, ampcor.secondaryImageWidth = sec_shape
    return ampcor


def _child_strip(ref_path: str, ref_shape: tuple, sec_path: str, sec_shape: tuple, params: dict, out_dir: str) -> None:
    """Estimate one strip of the offset grid, the same way dense_offsets.run does for the whole grid."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ampcor = _new_ampcor(ref_path, ref_shape, sec_path, sec_shape)
    ampcor = dense_offsets.set_optional_attributes(ampcor, params, *ref_shape)
    ampcor.offsetImageName = str(out / 'dense_offsets')
    ampcor.grossOffsetImageName = str(out / 'gross_offset')
    ampcor.snrImageName = str(out / 'snr')
    ampcor.covImageName = str(out / 'covariance')
    ampcor.corrImageName = str(out / 'correlation_peak')
    ampcor.runAmpcor()


def dense_offsets_in_strips(cfg: dict) -> None:
    """Drop-in for dense_offsets.run on CPU: the same windows, estimated in parallel azimuth strips."""
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    scratch_path = Path(cfg['product_path_group']['scratch_path'])
    offset_params = cfg['processing']['dense_offsets']
    lines_per_block = offset_params['lines_per_block']
    coregistered_slc_path = Path(offset_params['coregistered_slc_path'])
    ref_slc = SLC(hdf5file=ref_hdf5)
    t0 = time.time()
    for freq, _, pol_list in get_cfg_freq_pols(cfg):
        for pol in pol_list:
            out_dir = scratch_path / f'dense_offsets/freq{freq}' / pol
            out_dir.mkdir(parents=True, exist_ok=True)
            # one memory-mappable copy of the reference SLC, shared by every strip
            ref_path = str(out_dir / 'reference.slc')
            dense_offsets.copy_raster(ref_hdf5, freq, pol, lines_per_block, ref_path, file_type='ENVI')
            ref_raster = isce3.io.Raster(f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}')
            ref_shape = (ref_raster.length, ref_raster.width)
            if coregistered_slc_path.is_file():
                sec_path = str(out_dir / 'secondary.slc')
                dense_offsets.copy_raster(sec_hdf5, freq, pol, lines_per_block, sec_path, file_type='ENVI')
            else:
                sec_path = str(
                    coregistered_slc_path / f'coarse_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc'
                )
            sec_raster = isce3.io.Raster(sec_path)
            sec_shape = (sec_raster.length, sec_raster.width)

            # the full grid, from isce3's own parameter logic
            grid = _new_ampcor(ref_path, ref_shape, sec_path, sec_shape)
            grid = dense_offsets.set_optional_attributes(grid, offset_params, *ref_shape)
            n_down, n_across = grid.numberWindowDown, grid.numberWindowAcross
            n_strips = min(os.cpu_count(), n_down)
            rows = np.linspace(0, n_down, n_strips + 1).astype(int)
            children = []
            for k in range(n_strips):
                # this strip's rows of the same grid: shift the first window down, keep the range layout
                params = dict(
                    offset_params,
                    offset_width=n_across,
                    offset_length=int(rows[k + 1] - rows[k]),
                    start_pixel_range=grid.referenceStartPixelAcrossStatic,
                    start_pixel_azimuth=grid.referenceStartPixelDownStatic + int(rows[k]) * grid.skipSampleDown,
                )
                strip_dir = str(out_dir / f'strip_{k:03d}')
                child = SPAWN.Process(
                    target=_child_strip, args=(ref_path, ref_shape, sec_path, sec_shape, params, strip_dir)
                )
                child.start()
                children.append(child)
            for k, child in enumerate(children):
                _join(child, f'dense offsets strip {k}')

            # stitch: the rasters are flat float32 with pixel-interleaved bands, so strips concatenate in row order
            for name, bands in OFFSET_RASTERS.items():
                dense_offsets.create_empty_dataset(str(out_dir / name), n_across, n_down, bands, gdal.GDT_Float32)
                with (out_dir / name).open('wb') as whole:
                    for k in range(n_strips):
                        whole.write((out_dir / f'strip_{k:03d}' / name).read_bytes())
            for k in range(n_strips):
                shutil.rmtree(out_dir / f'strip_{k:03d}')
            log.info(f'dense offsets freq{freq} {pol}: grid {n_down} x {n_across} in {n_strips} strips')
    log.info(f'successfully ran dense_offsets in strips in {time.time() - t0:.3f} seconds')


# --- the unwrap-looks crossmul alongside the first crossmul ----------------------------------------------------------
def _child_crossmul(
    yaml_path: str, resample_type: str, rg_looks: int, az_looks: int, scratch: str, inputs: dict, slc_copies: dict
) -> None:
    """Run crossmul at the unwrap looks, writing its rasters to disk in a scratch folder of its own."""
    # link the SLC copies the parent already has instead of copying them again
    _SLC_COPIES.update(slc_copies)
    _reuse_slc_copies()
    cfg = _load_cfg(yaml_path)
    # bandpass_insar may have pointed the parent's runconfig at bandpassed RSLCs; read the same files.
    cfg['input_file_group'].update(inputs)
    # Own scratch folder, so its reference.slc copy does not collide with the first crossmul's. The coregistered
    # SLC and flattening paths were already resolved to the real scratch when the runconfig loaded.
    cfg['product_path_group']['scratch_path'] = scratch
    Path(scratch, 'crossmul').mkdir(parents=True, exist_ok=True)
    crossmul.run(
        cfg, output_hdf5=None, resample_type=resample_type, dump_on_disk=True, rg_looks=rg_looks, az_looks=az_looks
    )


def _overlap_crossmul(yaml_path: str) -> None:
    """Replace crossmul.run with one that starts the unwrap-looks crossmul early and hands it to unwrap."""
    state: dict[str, Any] = {}
    isce3_crossmul = crossmul.run

    def crossmul_run(
        cfg: dict,
        output_hdf5: str | None = None,
        resample_type: str = 'coarse',
        dump_on_disk: bool = False,
        rg_looks: int | None = None,
        az_looks: int | None = None,
    ) -> None:
        unwrap_cfg = cfg['processing']['phase_unwrap']
        looks = (unwrap_cfg['range_looks'], unwrap_cfg['azimuth_looks'])
        scratch = Path(cfg['product_path_group']['scratch_path'])
        # First top-level crossmul (writes the RIFG): start the unwrap-looks crossmul next to it.
        if 'child' not in state and not dump_on_disk and max(looks) > 1:
            state['looks'], state['scratch'] = looks, scratch
            child_scratch = str(scratch / 'crossmul_unwrap_looks')
            inputs = {key: cfg['input_file_group'][key] for key in ('reference_rslc_file', 'secondary_rslc_file')}
            child_args = (yaml_path, resample_type, *looks, child_scratch, inputs, dict(_SLC_COPIES))
            state['child'] = SPAWN.Process(target=_child_crossmul, args=child_args)
            state['child'].start()
            return isce3_crossmul(cfg, output_hdf5, resample_type, dump_on_disk, rg_looks, az_looks)
        # unwrap asks for the unwrap-looks crossmul: collect the child and put its rasters where unwrap reads them.
        if 'child' in state and not state.get('joined') and dump_on_disk and (rg_looks, az_looks) == state['looks']:
            state['joined'] = True
            _join(state['child'], 'crossmul')
            made = state['scratch'] / 'crossmul_unwrap_looks' / 'crossmul'
            for raster in made.glob(f'freq*/*/*_rg{rg_looks}_az{az_looks}*'):
                shutil.move(raster, state['scratch'] / 'crossmul' / raster.relative_to(made))
            shutil.rmtree(made.parent)
            log.info('unwrap-looks crossmul collected from its child process')
            return None
        # every other crossmul (the ionosphere's own) runs as isce3 wrote it
        return isce3_crossmul(cfg, output_hdf5, resample_type, dump_on_disk, rg_looks, az_looks)

    crossmul.run = crossmul_run


def resample_and_sum_offsets(
    culled_path: str,
    geo_path: str,
    out_path: str,
    off_az_pos: np.ndarray,
    off_rg_pos: np.ndarray,
    length: int,
    width: int,
    invalid_value: float,
    lines_per_block: int = 512,
) -> None:
    """Write one rubbersheet offsets raster for fine resample, on the full reference RSLC grid.

    isce3 does this in two passes: rubbersheet._resample_offsets_to_slc writes the culled offsets resampled
    to the RSLC grid, then helpers.sum_gdal_rasters reads that file back and adds the geo2rdr offsets. This is
    the same arithmetic in one pass, so the resampled raster is never written. On a full frame the output was
    byte-identical to isce3's (T120 F023, 2026-11-18).

    Args:
        culled_path: Culled (polynomial or interpolated) offsets raster, on the offsets grid.
        geo_path: geo2rdr offsets raster (flat float64), on the reference RSLC grid.
        out_path: Output ENVI raster, the sum on the reference RSLC grid.
        off_az_pos: Reference RSLC line (fractional) of each offsets grid row.
        off_rg_pos: Reference RSLC column (fractional) of each offsets grid column.
        length: Number of lines of the reference RSLC.
        width: Number of columns of the reference RSLC.
        invalid_value: Value marking invalid samples; invalid in either input gives invalid out.
        lines_per_block: Number of RSLC lines per block.
    """
    culled = gdal.Open(culled_path, gdal.GA_ReadOnly)
    off = culled.GetRasterBand(1).ReadAsArray()

    # fractional offsets-grid index and bilinear weight of each RSLC pixel, clamped to the grid edges
    def index_weight(pos: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.interp(np.arange(n), pos, np.arange(len(pos)))
        i0 = np.clip(np.floor(idx).astype(int), 0, max(len(pos) - 2, 0))
        i1 = np.minimum(i0 + 1, len(pos) - 1)
        return i0, i1, idx - i0

    c0, c1, wc = index_weight(off_rg_pos, width)
    r0, r1, wr = index_weight(off_az_pos, length)
    wc = wc[None, :]

    # the writer isce3's sum uses, so the raster and its header come out the same
    out = GDALRaster.create_dataset_file(
        filepath=Path(out_path), dtype=np.float64, shape=(length, width), num_bands=1, driver_name='ENVI'
    )
    # read front to back with plain file reads; the kernel reads ahead on these far better than on a memory map
    with Path(geo_path).open('rb') as geo_file:
        for start in range(0, length, lines_per_block):
            rows = slice(start, min(start + lines_per_block, length))
            w = wr[rows, None]
            # interpolate along azimuth, then along range (isce3's expressions, unchanged)
            off_az = off[r0[rows]] * (1 - w) + off[r1[rows]] * w
            block = off_az[:, c0] * (1 - wc) + off_az[:, c1] * wc
            geo_block = np.fromfile(geo_file, dtype=np.float64, count=block.size).reshape(block.shape)
            invalid = (geo_block == invalid_value) | (block == invalid_value)
            out[rows, :] = np.where(invalid, invalid_value, geo_block + block)


def _overlap_rubbersheet_rasters() -> None:
    """Replace rubbersheet's resample and sum passes so the azimuth and range rasters are made side by side.

    For each polarization rubbersheet calls resample then sum for azimuth, then the same for range. The
    resample stand-in only notes its arguments. The sum stand-in starts the azimuth raster in a child
    process and returns, then makes the range raster in this process and waits for the azimuth child, so
    both rasters exist when the range call returns.
    """
    state: dict[str, Any] = {}

    def resample_offsets_to_slc(
        off_path: str, out_path: str, off_az_pos: np.ndarray, off_rg_pos: np.ndarray, length: int, width: int
    ) -> None:
        state[out_path] = (off_path, off_az_pos, off_rg_pos, length, width)

    def sum_gdal_rasters(geo_path: str, resampled_path: str, out_path: str, invalid_value: float = np.nan) -> None:
        off_path, off_az_pos, off_rg_pos, length, width = state.pop(resampled_path)
        args = (off_path, geo_path, out_path, off_az_pos, off_rg_pos, length, width, invalid_value)
        azimuth = Path(out_path).name == 'azimuth.off'
        # isce3 asks for azimuth first, then range; anything else would leave a child unjoined
        if azimuth == ('child' in state):
            raise RuntimeError(f'rubbersheet asked for {out_path} out of the azimuth-then-range order')
        if azimuth:
            state['child'] = SPAWN.Process(target=resample_and_sum_offsets, args=args)
            state['child'].start()
            return
        resample_and_sum_offsets(*args)
        _join(state.pop('child'), 'rubbersheet azimuth offsets')

    rubbersheet._resample_offsets_to_slc = resample_offsets_to_slc
    rubbersheet.sum_gdal_rasters = sum_gdal_rasters


# === overlapped steps ================================================================================================
def _only_steps(*enabled: str) -> dict[str, bool]:
    """Build an insar.run step table with only the named steps switched on."""
    return {step: step in enabled for step in Persistence.insar_steps}


def _write_products(yaml_path: str, inputs: dict) -> None:
    """Create the RIFG, RUNW and GUNW product files with their metadata and empty datasets, in a side process."""
    cfg = _load_cfg(yaml_path)
    # bandpass_insar may have pointed the main process's runconfig at bandpassed RSLCs; use the same files.
    cfg['input_file_group'].update(inputs)
    prepare_insar_hdf5.run(cfg)


def _run_under_unwrap(yaml_path: str, inputs: dict) -> None:
    """Run the steps that need the wrapped interferogram but not the unwrapped one, in a side process.

    They write only the GUNW file, so they can run while the unwrap writes RUNW.
    """
    cfg = _load_cfg(yaml_path)
    cfg['input_file_group'].update(inputs)
    # insar.run would otherwise clear the scratch folders the unwrap is still using.
    cfg['worker']['intermediate_files_removal_enabled'] = False
    _, out_paths = h5_prep.get_products_and_paths(cfg)
    geocode_insar.run(cfg, out_paths['RIFG'], out_paths['GUNW'], InputProduct.RIFG)
    # insar.run applies the runconfig's own enabled checks to these two steps.
    insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('troposphere', 'solid_earth_tides'))


class _SideBandDone(Exception):
    """Raised in the side process once the ionosphere's side-band interferogram is made."""


def _child_side_band(yaml_path: str, inputs: dict, runw_hdf5: str) -> None:
    """Make the ionosphere's side-band (frequency B) interferogram in a side process, and stop there.

    ionosphere.run prepares its runconfig, has run_insar_workflow make the side-band interferogram in
    scratch/ionosphere/<method>, and only then reads the main unwrapped phase. Running isce3's own function up
    to the end of run_insar_workflow gives exactly the side-band products the ionosphere step would make.
    """
    cfg = _load_cfg(yaml_path)
    cfg['input_file_group'].update(inputs)
    isce3_workflow = ionosphere.run_insar_workflow

    def workflow_then_stop(iono_cfg: dict, original: dict, out_paths: dict, unwrapping_flag: bool = True) -> None:
        isce3_workflow(iono_cfg, original, out_paths, unwrapping_flag=unwrapping_flag)
        raise _SideBandDone

    ionosphere.run_insar_workflow = workflow_then_stop
    try:
        ionosphere.run(cfg, runw_hdf5)
    except _SideBandDone:
        return
    raise RuntimeError('the ionosphere step ran without making a side-band interferogram')


def start_side_band(cfg: dict, yaml_path: str, runw_hdf5: str) -> multiprocessing.process.BaseProcess | None:
    """Start the ionosphere's side-band interferogram in a side process, to overlap it with the unwrap.

    NOT called by fast_run: written from a reading of isce3 0.25.22's ionosphere.py and never run on a real
    pair (2026-11-18). On a full frame the side-band chain is about 410 s (frequency B rdr2geo, geo2rdr,
    product file, resample and crossmul) that needs the rubbersheet offsets and nothing from the main unwrap.
    To use it, call it after fine resample and before the unwrap step; the ionosphere step then collects it.

    Only the side-band methods are handled; the split-spectrum methods interleave their sub-band runs with
    other work, and are left to isce3.

    Args:
        cfg: isce3 InSAR runconfig dictionary of the main process.
        yaml_path: Path of the isce3 runconfig, which the side process reloads.
        runw_hdf5: Path of the main RUNW product (passed through to isce3; the side band does not open it).

    Returns:
        The side process, or None when there is no side-band interferogram to make.
    """
    iono = cfg['processing']['ionosphere_phase_correction']
    processed = cfg['processing']['input_subset']['list_of_frequencies']
    # isce3 makes a side-band interferogram only for polarizations the main run did not process
    extra = any(
        pol not in (processed.get(freq) or [])
        for freq, pols in iono['list_of_frequencies'].items()
        for pol in (pols or [])
    )
    if not iono['enabled'] or iono['spectral_diversity'] not in ('main_side_band', 'main_diff_ms_band') or not extra:
        return None

    inputs = {key: cfg['input_file_group'][key] for key in ('reference_rslc_file', 'secondary_rslc_file')}
    child = SPAWN.Process(target=_child_side_band, args=(yaml_path, inputs, runw_hdf5))
    child.start()
    isce3_workflow = ionosphere.run_insar_workflow

    def collect_side_band(iono_cfg: dict, original: dict, out_paths: dict, unwrapping_flag: bool = True) -> None:
        # the ionosphere step asks for the side-band interferogram: the side process has made it
        ionosphere.run_insar_workflow = isce3_workflow
        _join(child, 'ionosphere side band')
        log.info('ionosphere side-band interferogram collected from its side process')

    ionosphere.run_insar_workflow = collect_side_band
    return child


def _use_whirlwind(cfg: dict, runconfig_path: str) -> None:
    """Switch the phase unwrapper to whirlwind, for the main unwrap and the ionosphere's own.

    Set in the loaded runconfig and in the yaml: the child processes reload the yaml, the product
    writers record it in the products, and it is packaged with the GUNW.
    """
    cfg['processing']['phase_unwrap']['algorithm'] = 'whirlwind'
    runconfig = yaml.safe_load(Path(runconfig_path).read_text())
    runconfig['runconfig']['groups']['processing'].setdefault('phase_unwrap', {})['algorithm'] = 'whirlwind'
    Path(runconfig_path).write_text(yaml.safe_dump(runconfig, sort_keys=False))


def fast_run(cfg: dict, out_paths: dict, yaml_path: Path) -> None:
    """Run the isce3 InSAR workflow for one pair with the speedups in this module.

    A stand-in for ``insar.run(cfg, out_paths, run_steps)`` with every step on. It replaces functions
    inside isce3's modules, so the process that calls it should not go on to make a standard run.

    Args:
        cfg: isce3 InSAR runconfig dictionary, loaded from ``yaml_path``.
        out_paths: Paths of the RIFG, RUNW and GUNW product files.
        yaml_path: Path of the isce3 runconfig, which the child processes reload.
    """
    # absolute, since the children resolve it themselves
    runconfig_path = str(Path(yaml_path).resolve())
    _use_whirlwind(cfg, runconfig_path)

    # faster functions
    offsets_polyfit.polyfit_offsets = polyfit_offsets
    ionosphere_filter.nan_median_filter = nan_median_filter
    ionosphere_filter.convolve_preserve_nan = convolve_preserve_nan
    _reuse_slc_copies()

    # parallel steps; the strips drive CPU ampcor, so a GPU run keeps isce3's own dense offsets
    if not cfg['worker']['gpu_enabled']:
        dense_offsets.run = dense_offsets_in_strips
    _overlap_crossmul(runconfig_path)
    _overlap_rubbersheet_rasters()

    # insar.run clears scratch folders at the end of every call, whichever steps ran, so removal
    # is held off until the last call.
    worker = cfg['worker']
    remove_intermediates = worker['intermediate_files_removal_enabled']
    worker['intermediate_files_removal_enabled'] = False

    insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('bandpass_insar', 'rdr2geo', 'geo2rdr'))
    # the RSLCs the main process reads from here on (bandpass_insar may have replaced them)
    inputs = {key: cfg['input_file_group'][key] for key in ('reference_rslc_file', 'secondary_rslc_file')}

    # The product writers need geo2rdr's offsets and nothing later, and the files they create are
    # first opened by offsets_product or rubbersheet, so they run next to coarse resample and dense
    # offsets. Running them outside this process is what puts the dense offsets in their second
    # state (see the module docstring), so this overlap is the source of the product difference.
    with ProcessPoolExecutor(max_workers=1, mp_context=SPAWN) as pool:
        writers_job = pool.submit(_write_products, runconfig_path, inputs)
        insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('coarse_resample', 'dense_offsets'))
        writers_job.result()

    insar.run(
        cfg=cfg,
        out_paths=out_paths,
        run_steps=_only_steps('offsets_product', 'rubbersheet', 'fine_resample', 'crossmul', 'filter_interferogram'),
    )

    with ProcessPoolExecutor(max_workers=1, mp_context=SPAWN) as pool:
        side_job = pool.submit(_run_under_unwrap, runconfig_path, inputs)
        insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('unwrap'))
        side_job.result()

    insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('ionosphere'))
    geocode_insar.run(cfg, out_paths['RUNW'], out_paths['GUNW'], InputProduct.RUNW)

    # Last call: baseline, then the scratch clean-up the earlier calls skipped.
    worker['intermediate_files_removal_enabled'] = remove_intermediates
    insar.run(cfg=cfg, out_paths=out_paths, run_steps=_only_steps('baseline'))
