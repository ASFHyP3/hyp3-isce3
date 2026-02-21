
from hyp3_isce3.process import get_config


def test_get_config():
    reference_path = 'REFERENCE_TEST.h5'
    secondary_path = 'SECONDARY_TEST.h5'

    yaml = get_config(reference_path, secondary_path)
    exists_reference = False
    exists_secondary = False
    with yaml.open('r') as cfg:
        lines = cfg.readlines()
        for line in lines:
            if 'reference_rslc_file' in line:
                assert reference_path in line
                exists_reference = True
            if 'secondary_rslc_file' in line:
                assert secondary_path in line
                exists_secondary = True
    assert exists_reference and exists_secondary
