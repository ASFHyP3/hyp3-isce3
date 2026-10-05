# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1]

### Changed
- With `--subset`, the ECMWF troposphere weather models are now streamed and cropped to the AOI plus a 1-degree buffer (new `hyp3_isce3.crop_tropo` module) instead of downloading each ~2 GB global file. The cropped file is ~0.5 MB and gives identical RAiDER delays.

### Fixed
- `--subset` crops now derive the frequencyB range window from frequencyA's so both bands start at the same slant range. isce3's split-band ionosphere step decimates frequencyA offsets onto frequencyB, so independently windowed bands misregistered frequencyB by over a resolution cell, decorrelating it and leaving the ionosphere filter mask (bit 24 of the GUNW `mask` layer) speckled.
- `--subset` crops now snap their origin to both the crossmul and phase_unwrap looks (and frequencyB to its own unwrap looks), so every multilook grid lines up with a full-frame run.
- `--subset` crops now crop `inputDataExceptionMask` with the image window and set `identification/isFullFrame` to `False`.
- Corrected copy-pasted or outdated docstrings in `process.py` and `__main__.py`.

## [0.3.0]

> [!IMPORTANT]
> This release includes a major change to the hyp3-isce3 development environment! hyp3-isce3 now uses [pixi](https://pixi.sh/) to manage development environments.

### Added
- Added a `--subset LON_MIN LAT_MIN LON_MAX LAT_MAX` option that streams cropped input RSLCs to a WGS84 bounding box before processing, so the InSAR workflow runs on a small radar-coordinate patch (minutes instead of hours for a full frame) and the GUNW output is bounded to that area of interest.
- Added the `hyp3_isce3.crop_rslc` module, which maps the AOI into each RSLC's radar grid with isce3 `geo2rdr` and writes a cropped RSLC. The cropped product's identification times, `boundingPolygon`, geolocation grid, and `processingInformation` metadata cubes are kept consistent with the crop so the resulting GUNW metadata reflects the AOI.
- Added authentication function.

### Changed
- Changed `reference` and `secondary` parameters for `granules`.
- hyp3-isce3 now uses [pixi](https://pixi.sh/) to manage development environments instead of conda/mamba. For more info, see [#23](https://github.com/ASFHyP3/hyp3-isce3/pull/23).
  - Environments, their dependencies, etc. are now all configured in the `tool.pixi` sections of the `pyproject.toml`
  - Pixi now writes a `pixi.lock` file.
  - The Dockerfile uses the pixi base image and `entrypoint.sh` has been updated to use the pixi environment accordingly.
- hyp3-isce3 docker images are now multiarch, supporting linux-amd64 and linux-arm64. For more info, see [#23](https://github.com/ASFHyP3/hyp3-isce3/pull/23). 
- CI/CD pipelines that required an `environment.yml` have been reimplemented to use pixi, including `build.yml`, `static-analysis.yml` and `test.yml`.

### Removed
- Environment/requirements files used by conda/mamba, such as `environment.yml` and `requirements-*.txt`, in favor of pixi config and lock files.

## [0.2.0]

### Added
- Added ionospheric and tropospheric corrections.
- Added initial workflow to get a GUNW from a pair of NISAR images. The parameters `--reference` and `secondary` refer to the scene names of the RSLCs.

## [0.1.0]

### Added
- hyp3-isce3 plugin created with the [HyP3 Cookiecutter](https://github.com/ASFHyP3/hyp3-cookiecutter)
