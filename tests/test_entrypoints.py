def test_hyp3_isce3(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '-h'])
    assert ret.success
