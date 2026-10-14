import contextlib
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from nisar.mixed_mode import PolChannelSet

import hyp3_isce3
from hyp3_isce3.focus import inputs
from hyp3_isce3.focus.process import get_config, only_polarizations


L0B = 'NISAR_L0_PR_RRSD_032_128_D_161S_20261009T021028_20261009T021116_P05023_F_J_001'
FALLBACK = Path(hyp3_isce3.__file__).parent / 'schemas' / 'rslc_fallback.yaml'
ANCILLARIES = {
    'orbit': 'ORB.xml',
    'pointing': 'RP.xml',
    'external_calibration': 'EXT.yaml',
    'internal_calibration': 'INT.h5',
    'antenna_pattern': 'ANT.h5',
}


def test_decode_l0b_name():
    l0b = inputs.decode_l0b_name(f'/some/dir/{L0B}.h5')
    assert (l0b['cycle'], l0b['track'], l0b['direction'], l0b['crid']) == ('032', '128', 'D', 'P05023')
    assert l0b['start_time'] == datetime(2026, 10, 9, 2, 10, 28)
    assert l0b['end_time'] == datetime(2026, 10, 9, 2, 11, 16)


def test_get_config_production_template(tmp_path):
    # Production template: frame times and track-frame geometry pass through; job paths and identity are ours.
    template = Path(shutil.copy(FALLBACK, tmp_path / 'temp.yaml'))
    runconfig = yaml.safe_load(template.read_text())
    groups = runconfig['runconfig']['groups']
    groups['processing']['output_grid'].update(
        {'start_time': '2026-10-09T02:10:32Z', 'end_time': '2026-10-09T02:10:40Z'}
    )
    template.write_text(yaml.safe_dump(runconfig))

    with contextlib.chdir(tmp_path):
        yaml_path = get_config(template, [f'{L0B}.h5'], ANCILLARIES, 'dem.tif')
        groups = yaml.safe_load(yaml_path.read_text())['runconfig']['groups']

    assert groups['input_file_group']['input_file_path'] == [f'{L0B}.h5']
    anc = groups['dynamic_ancillary_file_group']
    assert anc['dem_file'] == 'dem.tif'
    assert all(anc[k] == v for k, v in ANCILLARIES.items())
    assert 'corner_reflector_file' not in anc
    assert anc['dem_file_description'].startswith('Digital Elevation Model')
    assert groups['product_path_group']['sas_output_file'] == 'output/rslc.h5'

    exe = groups['primary_executable']
    assert (exe['processing_type'], exe['processing_center']) == ('OD', 'A')
    assert exe['partial_granule_id'] == (
        'NISAR_L1_OD_RSLC_032_128_D_076_{MODE}_{POLE}_A_{StartDateTime}_{EndDateTime}_P05023_N_{C}_A_001'
    )
    assert groups['worker']['gpu_enabled'] is True
    assert groups['processing']['output_grid']['start_time'] == '2026-10-09T02:10:32Z'
    assert groups['geometry']['frame_number'] == 76


def test_get_template_fallback(tmp_path, monkeypatch):
    # No production RSLC: the shipped template gets this L0B's frame geometry and an OD granule ID.
    monkeypatch.setattr(inputs, 'find_production_runconfig', lambda name, frame: None)
    monkeypatch.setattr(inputs, 'get_frame_polygon', lambda track, frame: '{"type": "Polygon"}')

    with contextlib.chdir(tmp_path):
        template, is_production = inputs.get_template(L0B, 77)
        groups = yaml.safe_load(template.read_text())['runconfig']['groups']

    assert not is_production
    assert groups['input_file_group']['input_file_path'] == [f'{L0B}.h5']
    geometry = groups['geometry']
    assert (geometry['relative_orbit_number'], geometry['frame_number'], geometry['cycle_number']) == (128, 77, 32)
    assert geometry['orbit_direction'] == 'Descending'
    assert geometry['track_frame_polygon'] == '{"type": "Polygon"}'
    assert groups['primary_executable']['partial_granule_id'].startswith('NISAR_L1_OD_RSLC_032_128_D_077_')
    # The whole L0B is focused, so the frame-specific output times stay blank.
    assert groups['processing']['output_grid']['start_time'] is None
    assert groups['processing']['output_grid']['end_time'] is None


class FakeRaw:
    """Stands in for an L0B reader: dual-pol HH/HV in both frequencies, as on the T120 L0Bs."""

    polarizations = {'A': ['HH', 'HV'], 'B': ['HH', 'HV']}

    def getChirpParameters(self, freq_id, tx):
        # center frequency, sample rate, chirp slope, duration: 20 MHz on A, 5 MHz on B
        return (1.2575e9, 24e6, 20e6 / 40e-6, 40e-6) if freq_id == 'A' else (1.2875e9, 6e6, 5e6 / 40e-6, 40e-6)


def test_only_polarizations_keeps_hh_in_both_frequencies():
    with only_polarizations(['HH']):
        mode = PolChannelSet.from_raw(FakeRaw())
    assert sorted((c.freq_id, c.pol) for c in mode) == [('A', 'HH'), ('B', 'HH')]
    # the original reader comes back after the focus
    assert len(PolChannelSet.from_raw(FakeRaw())) == 4


def test_only_polarizations_none_keeps_everything():
    with only_polarizations(None):
        assert len(PolChannelSet.from_raw(FakeRaw())) == 4


def test_only_polarizations_missing_pol_raises():
    with only_polarizations(['VV']), pytest.raises(ValueError, match='VV'):
        PolChannelSet.from_raw(FakeRaw())
    assert len(PolChannelSet.from_raw(FakeRaw())) == 4
