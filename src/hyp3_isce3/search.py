"""Find NISAR products in CMR across their maturity collections."""

import earthaccess


# A NISAR product's maturity is baked into its CMR collection short-name, and granules are
# spread across stages at any given time: beta/commissioning granules stay in BETA, newer ones
# in PROVISIONAL, and final/validated ones drop the tag ('' -> operational). We can't know a
# given granule's maturity ahead of time, so lookups search every stage, newest processing first.
NISAR_MATURITIES = ('PROVISIONAL', 'BETA', '')  # '' == operational


def nisar_short_names(level: str, product: str) -> list[str]:
    """CMR collection short-names for a NISAR product across all maturity stages.

    Args:
        level: Processing level, e.g. 'L1' or 'L2'.
        product: Product type, e.g. 'RSLC' or 'GUNW'.

    Returns:
        The candidate collection short-names, one per maturity stage.
    """
    return [f'NISAR_{level}_{product}' + (f'_{m}' if m else '') + '_V1' for m in NISAR_MATURITIES]


def search_nisar(level: str, product: str, **query: str) -> list[earthaccess.DataGranule]:
    """Search each maturity collection of a NISAR product until one returns granules.

    Args:
        level: Processing level, e.g. 'L1' or 'L2'.
        product: Product type, e.g. 'RSLC' or 'GUNW'.
        **query: earthaccess.search_data filters, e.g. granule_name or readable_granule_name.

    Returns:
        The granules from the first collection with a match.
    """
    short_names = nisar_short_names(level, product)
    for short_name in short_names:
        # count=-1 (all results) is earthaccess's default, given here so the query keywords can't bind to it.
        results = earthaccess.search_data(count=-1, short_name=short_name, **query)
        if results:
            return results
    raise ValueError(f'No NISAR {level} {product} granule found for {query} in {short_names}')


def find_rslc(scene_name: str) -> earthaccess.DataGranule:
    """Find a NISAR RSLC granule by name, whichever maturity collection holds it."""
    return search_nisar('L1', 'RSLC', readable_granule_name=scene_name)[0]
