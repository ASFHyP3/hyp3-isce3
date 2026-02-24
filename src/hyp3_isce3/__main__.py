"""isce3 processing for HyP3."""

import logging
from argparse import ArgumentParser

from hyp3lib.aws import upload_file_to_s3

from hyp3_isce3.process import process_isce3


def main() -> None:
    """HyP3 entrypoint for hyp3_isce3."""
    parser = ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    # TODO: Your arguments here
    parser.add_argument('--reference', help='Name of the reference scene')
    parser.add_argument('--secondary', help='Name of the secondary scene')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO
    )

    product_file = process_isce3(
        reference_scene=args.reference,
        secondary_scene=args.secondary,
    )

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
