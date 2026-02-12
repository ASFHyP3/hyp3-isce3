#!/bin/bash --login
set -e
conda activate hyp3-isce3
exec python -um hyp3_isce3 "$@"
