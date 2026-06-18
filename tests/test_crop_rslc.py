"""Tests for hyp3_isce3.crop_rslc.

The geo2rdr-based window solve (`aoi_to_radar_window`) needs a real orbit, so it
is not unit-tested here; instead the pure index math (`_padded_slice`) is tested
directly and the writer (`crop_rslc`) against a synthetic RSLC. These tests don't
call isce3, though importing the module does (isce3/nisar are imported at top).
"""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from hyp3_isce3 import crop_rslc as crop_mod
from hyp3_isce3.crop_rslc import (
    _band,
    _bracket,
    _check_epoch_alignment,
    _copy_attrs,
    _mirror_except,
    _padded_slice,
    crop_rslc,
    crop_rslc_pair,
    get_polarizations,
)


_SWATHS = 'science/LSAR/RSLC/swaths'

# --- synthetic geometry -----------------------------------------------------
# Clean integer axes so np.searchsorted lands on exact indices.
N_LINES = 100  # full azimuth lines
N_RGA, N_RGB = 120, 15  # full range samples, freq A / B
SWATH_T = np.arange(N_LINES, dtype='f8')  # 0..99
SLANT_A = np.arange(N_RGA, dtype='f8')  # 0..119
SLANT_B = np.arange(N_RGB, dtype='f8') * 8.0  # 0,8,..,112 (coarse, like real B)
VALID_START, VALID_END = 10, 110  # validSamplesSubSwath1 row value

MARGIN = 4
# Radar-coord extent (az time 30-50 s, slant range 36-60 m) and the padded
# single-axis windows from _window_from_radar_coords:
#   az: searchsorted(SWATH_T,[30,50])=[30,50] -> +/-4 -> (26,54)
#   A : searchsorted(SLANT_A,[36,60])=[36,60] -> +/-4 -> (32,64)
AZ_LO, AZ_HI, RG_LO, RG_HI = 30.0, 50.0, 36.0, 60.0
# A crop-writer fixture window (the freqB slice here is arbitrary; crop_rslc just
# slices to whatever indices it is given).
EXPECTED_WINDOW = {'az': (26, 54), 'frequencyA': (32, 64), 'frequencyB': (4, 9)}
POLARIZATIONS = {'frequencyA': ['HH', 'HV'], 'frequencyB': ['HH', 'HV']}

# Cropped swath physical extent for EXPECTED_WINDOW: az SWATH_T[26..53]=26..53,
# range SLANT_A[32..63]=32..63. Epoch is 2025-01-01, so the restamped
# identification times are epoch + 26 s and + 53 s.
TIME_UNITS = 'seconds since 2025-01-01 00:00:00'
EXPECTED_START = '2025-01-01T00:00:26'
EXPECTED_END = '2025-01-01T00:00:53'

# geolocationGrid axes (coarse, like real products). The bracketed crop of the
# swath extent (az 26-53, range 32-63) onto these axes is indices [2:7] for both.
_GEOLOC = 'science/LSAR/RSLC/metadata/geolocationGrid'
N_H, N_AZG, N_RGG = 2, 10, 10
GRID_HEIGHTS = np.array([0.0, 1000.0])
GRID_AZ = np.arange(N_AZG, dtype='f8') * 10.0  # 0,10,..,90
GRID_RG = np.arange(N_RGG, dtype='f8') * 12.0  # 0,12,..,108
EXPECTED_GRID_SLICE = slice(2, 7)  # both axes bracket [lo, hi] to [2:7]


def _make_rslc(path):
    """Write a tiny but structurally-faithful synthetic RSLC."""
    with h5py.File(path, 'w') as f:
        f.attrs['mission_name'] = 'NISAR'  # root attr -> must be copied verbatim

        sw = f.create_group(_SWATHS)
        t = sw.create_dataset('zeroDopplerTime', data=SWATH_T)
        t.attrs['units'] = TIME_UNITS  # attr on a cropped axis -> must survive; also the epoch
        sw.create_dataset('zeroDopplerTimeSpacing', data=1.0)

        for fr, slant in (('frequencyA', SLANT_A), ('frequencyB', SLANT_B)):
            g = sw.create_group(fr)
            nrg = slant.size
            # Distinct per-(line,sample) values so slicing is verifiable.
            for k, pol in enumerate(('HH', 'HV')):
                img = (np.arange(N_LINES)[:, None] * 1000 + np.arange(nrg)[None, :] + k * 0.5j).astype('c8')
                g.create_dataset(pol, data=img, chunks=(8, min(8, nrg)), compression='gzip')
            g.create_dataset('slantRange', data=slant)
            g.create_dataset('slantRangeSpacing', data=float(slant[1] - slant[0]))
            vs = np.tile([VALID_START, VALID_END], (N_LINES, 1)).astype('u4')
            g.create_dataset('validSamplesSubSwath1', data=vs)
            g.create_dataset('listOfPolarizations', data=np.array([b'HH', b'HV']))

        # geolocationGrid: own coarse axes + 3-D [height, az, range] coordinate layers.
        gl = f.create_group(_GEOLOC)
        gl.create_dataset('heightAboveEllipsoid', data=GRID_HEIGHTS)
        gl.create_dataset('zeroDopplerTime', data=GRID_AZ)
        gl.create_dataset('slantRange', data=GRID_RG)
        coord = np.broadcast_to(GRID_AZ[None, :, None] + GRID_RG[None, None, :], (N_H, N_AZG, N_RGG))
        gl.create_dataset('coordinateX', data=coord.copy())
        gl.create_dataset('coordinateY', data=coord.copy())
        gl.create_dataset('epsg', data=np.int32(4326))

        # identification times + footprint -> must be restamped to the cropped extent.
        ident = f.create_group('science/LSAR/identification')
        ident.create_dataset('zeroDopplerStartTime', data=np.bytes_('2025-01-01T00:00:00'))
        ident.create_dataset('zeroDopplerEndTime', data=np.bytes_('2025-01-01T00:01:39'))
        ident.create_dataset(
            'boundingPolygon', data=np.bytes_('POLYGON ((-180 -90, 180 -90, 180 90, -180 90, -180 -90))')
        )

        # processingInformation/parameters: own grid axes, a 1-D az layer, a non-grid
        # 1-D array, and a nested per-frequency cube -> grid pieces cropped, rest kept.
        pp = f.create_group('science/LSAR/RSLC/metadata/processingInformation/parameters')
        pp.create_dataset('zeroDopplerTime', data=GRID_AZ)
        pp.create_dataset('slantRange', data=GRID_RG)
        pp.create_dataset('referenceTerrainHeight', data=np.arange(N_AZG, dtype='f4'))  # az-indexed
        pp.create_dataset('rangeChirpWeighting', data=np.ones(8, dtype='f4'))  # not az/range -> verbatim
        fa = pp.create_group('frequencyA')
        fa.create_dataset('zeroDopplerTime', data=GRID_AZ)
        fa.create_dataset('slantRange', data=GRID_RG)
        fa.create_dataset('dopplerCentroid', data=np.outer(GRID_AZ, GRID_RG))  # (az, range)

        # A metadata dataset on no cropped axis -> must be copied byte-for-byte.
        orb = f.create_dataset(
            'science/LSAR/RSLC/metadata/orbit/position', data=np.arange(15, dtype='f8').reshape(5, 3)
        )
        orb.attrs['description'] = 'ECEF position'


@pytest.fixture
def rslc_h5(tmp_path):
    path = tmp_path / 'synthetic_rslc.h5'
    _make_rslc(path)
    return path


class TestWindowMath:
    """The pure extent -> padded index slice logic, isce3-free."""

    def test_az_and_range_windows_match_hand_derived(self):
        assert _padded_slice(SWATH_T, AZ_LO, AZ_HI, MARGIN) == (26, 54)
        assert _padded_slice(SLANT_A, RG_LO, RG_HI, MARGIN) == (32, 64)

    def test_partial_overlap_is_clamped(self):
        # Extent starts before the axis -> low bound clamps to 0.
        assert _padded_slice(SLANT_A, -50.0, RG_HI, MARGIN)[0] == 0

    def test_no_overlap_raises(self):
        # Extent entirely past the end of the axis.
        with pytest.raises(ValueError, match='does not overlap'):
            _padded_slice(SWATH_T, 1e6, 2e6, MARGIN)

    def test_non_monotonic_axis_raises(self):
        # searchsorted would silently return garbage on an unsorted axis.
        with pytest.raises(ValueError, match='increasing'):
            _padded_slice(np.array([0.0, 2.0, 1.0, 3.0]), 0.0, 3.0, MARGIN)


class TestCropRslc:
    def test_shapes_and_axes(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        a0, a1 = EXPECTED_WINDOW['az']
        with h5py.File(dst, 'r') as f:
            # Shared azimuth axis sliced once.
            assert f[f'{_SWATHS}/zeroDopplerTime'].shape == (a1 - a0,)
            np.testing.assert_array_equal(f[f'{_SWATHS}/zeroDopplerTime'][()], SWATH_T[a0:a1])
            for fr, slant in (('frequencyA', SLANT_A), ('frequencyB', SLANT_B)):
                p0, p1 = EXPECTED_WINDOW[fr]
                assert f[f'{_SWATHS}/{fr}/HH'].shape == (a1 - a0, p1 - p0)
                assert f[f'{_SWATHS}/{fr}/HV'].shape == (a1 - a0, p1 - p0)
                np.testing.assert_array_equal(f[f'{_SWATHS}/{fr}/slantRange'][()], slant[p0:p1])

    def test_image_content_is_the_window(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        a0, a1 = EXPECTED_WINDOW['az']
        p0, p1 = EXPECTED_WINDOW['frequencyA']
        with h5py.File(rslc_h5, 'r') as src, h5py.File(dst, 'r') as out:
            expect = src[f'{_SWATHS}/frequencyA/HH'][a0:a1, p0:p1]
            np.testing.assert_array_equal(out[f'{_SWATHS}/frequencyA/HH'][()], expect)

    def test_valid_samples_shifted_and_clipped(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        a0, a1 = EXPECTED_WINDOW['az']
        p0, p1 = EXPECTED_WINDOW['frequencyA']
        with h5py.File(dst, 'r') as f:
            vs = f[f'{_SWATHS}/frequencyA/validSamplesSubSwath1'][()]
        assert vs.shape == (a1 - a0, 2)
        # start = clip(10 - 32, 0, 32) = 0 ; end = clip(110 - 32, 0, 32) = 32
        expect = np.clip(np.array([VALID_START, VALID_END]) - p0, 0, p1 - p0)
        np.testing.assert_array_equal(vs[0], expect)
        assert (vs == expect).all()

    def test_metadata_and_attrs_copied_verbatim(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        with h5py.File(rslc_h5, 'r') as src, h5py.File(dst, 'r') as out:
            # Root attribute preserved.
            assert out.attrs['mission_name'] == src.attrs['mission_name']
            # Uncropped metadata dataset copied byte-for-byte, with its attrs.
            o = out['science/LSAR/RSLC/metadata/orbit/position']
            np.testing.assert_array_equal(o[()], src['science/LSAR/RSLC/metadata/orbit/position'][()])
            assert o.attrs['description'] == 'ECEF position'
            # Attribute on a cropped axis survives.
            assert out[f'{_SWATHS}/zeroDopplerTime'].attrs['units'] == TIME_UNITS

    def test_image_compression_preserved(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        with h5py.File(dst, 'r') as f:
            ds = f[f'{_SWATHS}/frequencyA/HH']
            assert ds.compression == 'gzip'
            assert ds.chunks is not None

    def test_geolocation_grid_bracketed(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        with h5py.File(dst, 'r') as f:
            # Own axes bracketed to span the cropped swath extent.
            np.testing.assert_array_equal(f[f'{_GEOLOC}/zeroDopplerTime'][()], GRID_AZ[EXPECTED_GRID_SLICE])
            np.testing.assert_array_equal(f[f'{_GEOLOC}/slantRange'][()], GRID_RG[EXPECTED_GRID_SLICE])
            # 3-D coordinate layer sliced on its trailing (az, range) dims only.
            assert f[f'{_GEOLOC}/coordinateX'].shape == (N_H, 5, 5)
            # Height axis and epsg are not on the radar grid -> untouched.
            np.testing.assert_array_equal(f[f'{_GEOLOC}/heightAboveEllipsoid'][()], GRID_HEIGHTS)
            assert f[f'{_GEOLOC}/epsg'][()] == 4326

    def test_identification_times_restamped(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        with h5py.File(dst, 'r') as f:
            assert f['science/LSAR/identification/zeroDopplerStartTime'][()].decode() == EXPECTED_START
            assert f['science/LSAR/identification/zeroDopplerEndTime'][()].decode() == EXPECTED_END

    def test_processing_parameters_cubes_cropped(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        pp = 'science/LSAR/RSLC/metadata/processingInformation/parameters'
        with h5py.File(dst, 'r') as f:
            # Own grid axes + the 1-D az layer bracketed to [2:7].
            np.testing.assert_array_equal(f[f'{pp}/zeroDopplerTime'][()], GRID_AZ[EXPECTED_GRID_SLICE])
            np.testing.assert_array_equal(f[f'{pp}/slantRange'][()], GRID_RG[EXPECTED_GRID_SLICE])
            assert f[f'{pp}/referenceTerrainHeight'].shape == (5,)
            # Nested per-frequency dopplerCentroid cropped on both az and range.
            assert f[f'{pp}/frequencyA/dopplerCentroid'].shape == (5, 5)
            # A non-grid 1-D array (chirp weighting) is left untouched.
            assert f[f'{pp}/rangeChirpWeighting'].shape == (8,)

    def test_bounding_polygon_recomputed(self, rslc_h5, tmp_path):
        dst = tmp_path / 'out.h5'
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW, POLARIZATIONS)
        with h5py.File(dst, 'r') as f:
            wkt = f['science/LSAR/identification/boundingPolygon'][()].decode()
        assert wkt.startswith('POLYGON ((') and '-180' not in wkt  # full-scene placeholder replaced
        # Cropped geolocationGrid corners ([2:7] on both axes), coordinate = GRID_AZ + GRID_RG.
        assert '44.000000 44.000000' in wkt  # (az 20, rg 24)
        assert '132.000000 132.000000' in wkt  # (az 60, rg 72)


class TestCropPair:
    def test_pair_writes_two_files(self, rslc_h5, tmp_path, monkeypatch):
        # The NISAR reader + geo2rdr window solve need isce3; stub the reader and
        # both reader-using steps so the pair plumbing (per-file crop + naming) is testable.
        monkeypatch.setattr(crop_mod, 'SLC', lambda **k: None)
        monkeypatch.setattr(crop_mod, 'aoi_to_radar_window', lambda *a, **k: EXPECTED_WINDOW)
        monkeypatch.setattr(crop_mod, 'get_polarizations', lambda *a, **k: POLARIZATIONS)
        # A real pair is two distinct files (different acquisition dates);
        # mimic that so the per-file output names don't collide.
        sec = tmp_path / 'synthetic_rslc_sec.h5'
        _make_rslc(sec)
        # bbox_wgs84/dem_file are unused here (the window solve is stubbed above),
        # but pass type-valid values so static analysis is happy.
        ref_sub, sec_sub = crop_rslc_pair(
            rslc_h5, sec, bbox_wgs84=[0.0, 0.0, 0.0, 0.0], dem_file='', out_dir=tmp_path / 'subs', margin=MARGIN
        )
        for p in (ref_sub, sec_sub):
            assert Path(p).exists()
            with h5py.File(p, 'r') as f:
                a0, a1 = EXPECTED_WINDOW['az']
                assert f[f'{_SWATHS}/zeroDopplerTime'].shape == (a1 - a0,)
        assert ref_sub != sec_sub


class TestBand:
    def test_returns_lsar(self, tmp_path):
        path = tmp_path / 'b.h5'
        with h5py.File(path, 'w') as f:
            f.create_group('science/LSAR/RSLC')
        with h5py.File(path, 'r') as f:
            assert _band(f) == 'LSAR'

    def test_returns_ssar(self, tmp_path):
        path = tmp_path / 'b.h5'
        with h5py.File(path, 'w') as f:
            f.create_group('science/SSAR/RSLC')
        with h5py.File(path, 'r') as f:
            assert _band(f) == 'SSAR'

    def test_no_rslc_band_raises(self, tmp_path):
        path = tmp_path / 'b.h5'
        with h5py.File(path, 'w') as f:
            f.create_group('science/LSAR/GCOV')  # a band group, but not RSLC
        with h5py.File(path, 'r') as f, pytest.raises(ValueError, match='No RSLC band'):
            _band(f)


class TestBracket:
    def test_brackets_extent_one_node_outside(self):
        axis = np.arange(10, dtype='f8') * 10.0  # 0, 10, ..., 90
        # searchsorted(right, 26)=3 -> 2 ; searchsorted(left, 53)=6 -> 7
        assert _bracket(axis, 26.0, 53.0) == (2, 7)

    def test_clamps_to_axis_bounds(self):
        axis = np.arange(10, dtype='f8') * 10.0
        assert _bracket(axis, -100.0, 1000.0) == (0, 10)


class TestEpochAlignment:
    def test_aligned_ok(self):
        # sensing_start == swath_t[0] -> no raise.
        _check_epoch_alignment(SimpleNamespace(sensing_start=0.0), np.arange(100, dtype='f8'))

    def test_within_half_sample_ok(self):
        # Offset below half an azimuth sample (spacing 1.0) -> no raise.
        _check_epoch_alignment(SimpleNamespace(sensing_start=0.4), np.arange(100, dtype='f8'))

    def test_misaligned_raises(self):
        with pytest.raises(RuntimeError, match='epoch'):
            _check_epoch_alignment(SimpleNamespace(sensing_start=5.0), np.arange(100, dtype='f8'))


class TestGetPolarizations:
    def test_maps_each_frequency_to_its_pols(self):
        slc = SimpleNamespace(frequencies=['A', 'B'], polarizations={'A': ['HH', 'HV'], 'B': ['VV']})
        assert get_polarizations(slc) == {'frequencyA': ['HH', 'HV'], 'frequencyB': ['VV']}


def test_copy_attrs(tmp_path):
    path = tmp_path / 'a.h5'
    with h5py.File(path, 'w') as f:
        src = f.create_dataset('src', data=[1, 2, 3])
        src.attrs['units'] = 'meters'
        src.attrs['scale'] = 2.0
        dst = f.create_dataset('dst', data=[0])
        _copy_attrs(src, dst)
        assert dst.attrs['units'] == 'meters'
        assert dst.attrs['scale'] == 2.0


def test_mirror_except_prunes_subtree(tmp_path):
    src_path, dst_path = tmp_path / 'src.h5', tmp_path / 'dst.h5'
    with h5py.File(src_path, 'w') as f:
        f.attrs['mission'] = 'NISAR'  # root attr -> copied
        f.create_dataset('a/keep', data=[1, 2])  # sibling of the pruned subtree
        f.create_dataset('a/drop/x', data=[9])  # inside the pruned subtree
        f.create_dataset('top', data=[7])  # unrelated branch
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        _mirror_except(src, dst, {'a/drop'})
    with h5py.File(dst_path, 'r') as out:
        assert out.attrs['mission'] == 'NISAR'
        assert 'a' in out  # ancestor of the pruned path is still created
        assert 'a/keep' in out  # sibling copied verbatim
        assert 'top' in out  # unrelated branch copied wholesale
        assert 'a/drop' not in out  # pruned subtree skipped for the caller to fill
