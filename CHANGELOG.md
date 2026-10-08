# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added
- `schemas/gunw_fallback.yaml`, a production GUNW runconfig (40+5 MHz parameter set) with its frame-specific output grid blanked. It is used when no production GUNW exists on the requested frame; the output projection is then the scene's UTM zone and isce3 sizes the grid from the RSLC.

### Changed
- `--subset` products set the GUNW name's coverage field to `P` (partial frame), matching `identification/isFullFrame`.
- The isce3 runconfig is now built from the production GUNW runconfig itself, replacing values only where our run differs (input/ancillary paths, output locations, ASF on-demand identity, subset grid). Products now carry production's composite release ID, product versions, and DOIs, and a granule ID for the actual pair (previously a stale example identity, e.g. `009_044_D_059_010`, `X05010`, BETA DOIs).
- Output products are named by the granule ID isce3 writes into the GUNW, following the NISAR GUNW filename convention (e.g. `..._005_019_A_011_006_4000_SH_..._P05023_N_F_A_001`).
- Each pair is processed in its own `./<product_id>/` working directory; `process_isce3` returns the absolute path of the zip inside it.
- NISAR RSLC and GUNW lookups search the PROVISIONAL, BETA, and operational collections through one shared `hyp3_isce3.search` module; full RSLC downloads now use earthaccess like streaming.
- Cropped RSLCs are named as their granule (`<scene>.h5`), so the GUNW's input granule metadata lists the true RSLC names.
- `aoi_to_radar_window` and `crop_streamed` now require `az_looks` and `rg_looks` as keyword arguments (`rg_looks_b` stays optional).

### Fixed
- `get_dem` mosaics only the EPSG:4326 NISAR DEM tiles. Near the poles the search also returns polar-stereographic (EPSG:3413/3031) copies of the same DEM, which `gdalbuildvrt` cannot mix, leaving a polar mosaic of the wrong extent (a 177 GB warp for an Iceland AOI).
- The GUNW product ID relabels only the producer field (`J` -> `A`), not every `J` in the trailing fields.
- The reference/secondary pair is ordered by parsed acquisition time inside `process_isce3`.

### Removed
- `schemas/insar.yaml` (the runconfig header template).

## [0.3.1]

### Changed
- With `--subset`, the ECMWF troposphere weather models are now streamed and cropped to the AOI plus a 1-degree buffer (new `hyp3_isce3.crop_tropo` module) instead of downloading each ~2 GB global file. The cropped file is ~0.5 MB and gives identical RAiDER delays.

### Fixed
- `--subset` crops now derive the frequencyB range window from frequencyA's so both bands start at the same slant range. isce3's split-band ionosphere step decimates frequencyA offsets onto frequencyB, so independently windowed bands misregistered frequencyB by over a resolution cell, decorrelating it and leaving the ionosphere filter mask (bit 24 of the GUNW `mask` layer) speckled.
- `--subset` crops now snap their origin to both the crossmul and phase_unwrap looks (and frequencyB to its own unwrap looks), so every multilook grid lines up with a full-frame run.
- `--subset` crops now crop `inputDataExceptionMask` with the image window and set `identification/isFullFrame` to `False`.
- Corrected copy-pasted or outdated docstrings in `process.py` and `__main__.py`.
- Updated ISCE3 version and included provisional collection for GUNWs.

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
