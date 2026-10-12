#!/bin/bash --login
set -e
pixi run -e gpu python -m hyp3_isce3 "$@"
