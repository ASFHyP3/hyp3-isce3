from hyp3_isce3.search import nisar_short_names


def test_nisar_short_names():
    # Newest processing stage first, then BETA, then the untagged operational collection.
    assert nisar_short_names('L2', 'GUNW') == [
        'NISAR_L2_GUNW_PROVISIONAL_V1',
        'NISAR_L2_GUNW_BETA_V1',
        'NISAR_L2_GUNW_V1',
    ]
    assert nisar_short_names('L1', 'RSLC')[0] == 'NISAR_L1_RSLC_PROVISIONAL_V1'
