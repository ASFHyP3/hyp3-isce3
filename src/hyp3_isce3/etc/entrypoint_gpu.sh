#!/bin/bash --login
set -e
# The gpu environment holds only the focus workflow's dependencies, so select it up front
# (the default, gunw, would fail to import).
pixi run -e gpu python -m hyp3_isce3 ++process focus_l0b "$@"
