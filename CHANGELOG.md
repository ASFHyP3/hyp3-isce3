# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added
- Added a `--subset LON_MIN LAT_MIN LON_MAX LAT_MAX` option that streams cropped input RSLCs to a WGS84 bounding box before processing, so the InSAR workflow runs on a small radar-coordinate patch (minutes instead of hours for a full frame) and the GUNW output is bounded to that area of interest.
- Added the `hyp3_isce3.crop_rslc` module, which maps the AOI into each RSLC's radar grid with isce3 `geo2rdr` and writes a cropped RSLC. The cropped product's identification times, `boundingPolygon`, geolocation grid, and `processingInformation` metadata cubes are kept consistent with the crop so the resulting GUNW metadata reflects the AOI.
- Added authentication function.

### Changed
- Changed `reference` and `secondary` parameters for `granules`.

## [0.2.0]

### Added
- Added ionospheric and tropospheric corrections.
- Added initial workflow to get a GUNW from a pair of NISAR images. The parameters `--reference` and `secondary` refer to the scene names of the RSLCs.

## [0.1.0]

### Added
- hyp3-isce3 plugin created with the [HyP3 Cookiecutter](https://github.com/ASFHyP3/hyp3-cookiecutter)
