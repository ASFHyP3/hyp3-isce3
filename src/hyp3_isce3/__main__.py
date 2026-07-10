"""isce3 processing for HyP3."""

import json
import logging
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.fetch import write_credentials_to_netrc_file

from hyp3_isce3.process import process_isce3


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


def nullable_json_dict(override_string: str) -> dict | None:
    """Parse the --override JSON string into a dict, treating empty/'None' as no overrides.

    HyP3 stringifies a null parameter to the literal 'None' before it reaches the command,
    so an unset override arrives as 'None' rather than being omitted.

    Args:
        override_string: JSON object of runconfig overrides, or '' / 'None' for no overrides.

    Returns:
        overrides: Parsed override dict, or None when no overrides were given.
    """
    override_string = override_string.strip()
    if override_string in ('', 'None'):
        return None
    return json.loads(override_string)


def earlier_granule_first(g1: str, g2: str) -> tuple[str, str]:
    """Sort granules for reference and secondary.

    Args:
        g1: First granule.
        g2: Second grnaule.

    Returns:
        reference: Reference granule.
        secondary: Secondary granule.
    """
    if g1.split('_')[11] <= g2.split('_')[11]:
        return g1, g2
    return g2, g1


def main() -> None:
    """HyP3 entrypoint for hyp3_isce3."""
    parser = ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    # TODO: Your arguments here
    parser.add_argument(
        'granules',
        type=str.split,
        nargs='+',
        help='NISAR RSLC granules',
    )
    parser.add_argument(
        '--subset',
        type=nullable_subset_list,
        nargs='*',
        help='Optional WGS84 bounding box to subset the output GUNW (LON_MIN LAT_MIN LON_MAX LAT_MAX)',
    )
    parser.add_argument(
        '--override',
        type=nullable_json_dict,
        default=None,
        help='Optional JSON of runconfig overrides under `groups`, e.g. '
        '\'{"processing.crossmul.range_looks": 11}\'. Only existing keys may be set.',
    )

    args = parser.parse_args()

    args.granules = [item for sublist in args.granules for item in sublist]
    if len(args.granules) != 2:
        parser.error('Must provide exactly two granules')

    if args.subset is not None:
        args.subset = [float(item) for sublist in args.subset for item in sublist]

        if len(args.subset) == 0:
            args.subset = None
        elif not len(args.subset) == 4:
            raise ValueError('The number of coordinates is not four')

    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    if username and password:
        write_credentials_to_netrc_file(username, password, append=False)

    if not (Path.home() / '.netrc').exists():
        warnings.warn(
            'Earthdata credentials must be present as environment variables, or in your netrc.',
            UserWarning,
        )

    reference_granule, secondary_granule = earlier_granule_first(*args.granules)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO
    )

    product_file = process_isce3(
        reference_scene=reference_granule,
        secondary_scene=secondary_granule,
        subset=args.subset,
        overrides=args.override,
    )

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
