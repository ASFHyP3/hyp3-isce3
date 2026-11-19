def test_hyp3_isce3(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '-h'])
    assert ret.success


def test_gunw(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '++process', 'gunw', '-h'])
    assert ret.success


def test_focus_l0b(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_isce3', '++process', 'focus_l0b', '-h'])
    assert ret.success


def test_numpy_trapz_available():
    # RAiDER 0.5.3 calls np.trapz; importing the package must provide it on any numpy.
    import numpy as np

    import hyp3_isce3  # noqa: F401

    assert np.trapz([1.0, 2.0, 4.0], x=[0.0, 1.0, 3.0]) == np.trapezoid([1.0, 2.0, 4.0], x=[0.0, 1.0, 3.0])
