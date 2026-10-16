"""isce3 processing for HyP3."""

import logging
import os
import sys
import warnings
from argparse import ArgumentParser
from importlib.metadata import entry_points
from pathlib import Path

from hyp3lib.fetch import write_credentials_to_netrc_file


def main() -> None:
    """HyP3 entrypoint for hyp3_isce3: dispatch to the workflow named by ++process."""
    # '+' prefix so the workflow's own '-'/'--' arguments (including -h) pass through untouched.
    parser = ArgumentParser(prefix_chars='+', add_help=False)
    parser.add_argument(
        '++process',
        choices=['gunw', 'focus_l0b'],
        default='gunw',
        help='Select the workflow to run: GUNW from two RSLCs (default) or RSLC focused from an L0B',
    )
    args, unknowns = parser.parse_known_args()

    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    if username and password:
        write_credentials_to_netrc_file(username, password, append=False)

    if not (Path.home() / '.netrc').exists():
        warnings.warn(
            'Earthdata credentials must be present as environment variables, or in your netrc.',
            UserWarning,
        )

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO
    )

    # Hand the remaining arguments to the selected workflow's own parser.
    (process_entry_point,) = entry_points(group='hyp3', name=args.process)
    sys.argv = [args.process, *unknowns]
    process_entry_point.load()()


if __name__ == '__main__':
    main()
