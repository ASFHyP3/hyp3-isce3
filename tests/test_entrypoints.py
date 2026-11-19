def test_hyp3_isce3(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '-h'])
    assert ret.success


def test_gunw(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '++process', 'gunw', '-h'])
    assert ret.success


def test_focus_l0b(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '++process', 'focus_l0b', '-h'])
    assert ret.success
