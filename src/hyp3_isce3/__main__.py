"""isce3 processing for HyP3."""

import logging
from argparse import ArgumentParser

from hyp3lib.aws import upload_file_to_s3

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


def main() -> None:
    """HyP3 entrypoint for hyp3_isce3."""
    parser = ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    # TODO: Your arguments here
    parser.add_argument('--reference', help='Name of the reference scene')
    parser.add_argument('--secondary', help='Name of the secondary scene')
    parser.add_argument(
        '--subset',
        type=nullable_subset_list,
        nargs='+',
        help='Optional WGS84 bounding box to subset the output GUNW (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)',
    )

    args = parser.parse_args()

    args.subset = [float(item) for sublist in args.subset for item in sublist]

    if len(args.subset) == 0:
        args.subset = None
    elif not len(args.subset) == 4:
        raise ValueError('The number of coordinates is not four')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO
    )

    product_file = process_isce3(
        reference_scene=args.reference,
        secondary_scene=args.secondary,
        subset=args.subset,
    )

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
