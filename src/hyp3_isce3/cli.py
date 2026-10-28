"""Argument parsing shared by the HyP3 entry points."""


def nullable_subset_list(subset_string: str) -> list[str]:
    """Returns a list of the subset.

    Args:
        subset_string: input string with the subset coordinates.

    Returns:
        subset_list: List of coordinates.
    """
    subset_string = subset_string.replace('None', '').strip()
    subset_list = [coord for coord in subset_string.split(' ') if coord]
    return subset_list


def parse_subset(subset: list[list[str]] | None) -> list[float] | None:
    """Flatten a parsed --subset argument to its four coordinates, or None when absent or blank.

    Args:
        subset: The argparse value of --subset (type nullable_subset_list, nargs '*').

    Returns:
        [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], or None.
    """
    if subset is None:
        return None
    values = [float(item) for sublist in subset for item in sublist]
    if not values:
        return None
    if len(values) != 4:
        raise ValueError('The number of coordinates is not four')
    return values
