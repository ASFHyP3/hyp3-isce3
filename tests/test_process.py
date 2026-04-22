from pathlib import Path

from hyp3_isce3.process import get_config


def test_get_config(mocker):
    reference_path = 'REFERENCE_TEST.h5'
    secondary_path = 'SECONDARY_TEST.h5'

    reference_orbit = 'REFERENCE_ORBIT.xml'
    secondary_orbit = 'SECONDARY_ORBIT.xml'

    reference_tropo = 'REFERENCE_TROPO.nc'
    secondary_tropo = 'SECONDARY_TROPO.nc'

    tec_path = 'TEC.json'
    watermask = 'WATERMASK.vrt'

    tmp_path = Path('temp.yaml')
    with tmp_path.open('w') as tmp:
        tmp.write('partial_granule_id:')

    mock_func = mocker.patch('hyp3_isce3.process.download_yaml')
    mock_func.return_value = tmp_path

    yaml = get_config(
        reference_path,
        secondary_path,
        reference_orbit,
        secondary_orbit,
        reference_tropo,
        secondary_tropo,
        tec_path,
        watermask,
    )
    exists_reference = False
    exists_secondary = False
    exists_orbits = False
    exists_tropo = False
    exists_tec = False
    exists_watermask = False
    with yaml.open('r') as cfg:
        lines = cfg.readlines()
        for line in lines:
            if 'reference_rslc_file' in line:
                assert reference_path in line
                exists_reference = True
            if 'secondary_rslc_file' in line:
                assert secondary_path in line
                exists_secondary = True
            if 'water_mask_file' in line:
                assert watermask in line
                exists_watermask = True
            if 'reference_orbit_file' in line or 'secondary_orbit_file' in line:
                assert reference_orbit in line or secondary_orbit in line
                exists_orbits = True
            if 'reference_troposphere_file' in line or 'secondary_troposphere_file' in line:
                assert reference_tropo in line or secondary_tropo in line
                exists_tropo = True
            if 'tec_file' in line:
                assert tec_path in line
                exists_tec = True
    assert exists_reference and exists_secondary and exists_watermask and exists_orbits and exists_tropo and exists_tec
    assert 'partial_granule_id:' in lines[-1]
