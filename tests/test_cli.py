import pytest

from hyp3_isce3.cli import nullable_subset_list, parse_subset


def test_nullable_subset_list():
    assert nullable_subset_list('None') == []
    assert nullable_subset_list('-117.4 33.3 -117.2 33.5') == ['-117.4', '33.3', '-117.2', '33.5']


def test_parse_subset():
    assert parse_subset(None) is None
    assert parse_subset([[]]) is None
    assert parse_subset([['-117.4', '33.3'], ['-117.2', '33.5']]) == [-117.4, 33.3, -117.2, 33.5]
    with pytest.raises(ValueError, match='four'):
        parse_subset([['1', '2', '3']])
