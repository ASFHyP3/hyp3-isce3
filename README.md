# HyP3 ISCE3 Plugin

The HyP3-ISCE3 plugin provides a workflow to process SAR satellite data using the [InSAR Scientific Computing Environment 3](https://github.com/isce-framework/isce3) (ISCE3) software package. This plugin is part of the [Alaska Satellite Facility's](https://asf.alaska.edu) larger HyP3 (Hybrid Plugin Processing Pipeline) system, which is a batch processing pipeline designed for on-demand processing of SAR data.

## Usage

The HyP3-ISCE3 plugin provides a workflow (accessible directly in Python or via a CLI) for creating NISAR geocoded unwrapped interferogram using ISCE3 workflow.

To run the workflow:

```
python -m hyp3_isce3 \
  NISAR_L1_PR_RSLC_005_172_A_008_2005_DHDH_A_20251122T024618_20251122T024652_X05007_N_F_J_001 \
  NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_20251204T024653_X05007_N_F_J_001 
```

The processing time can be reduced by using the `subset` parameter and a bounding box in the format `LON_MIN LAT_MIN LON_MAX LAT_MAX`:

```
python -m hyp3_isce3 \
  NISAR_L1_PR_RSLC_005_172_A_008_2005_DHDH_A_20251122T024618_20251122T024652_X05007_N_F_J_001 \
  NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_20251204T024653_X05007_N_F_J_001 \
  --subset 40.55 13.46 40.78 13.65
```

### Earthdata Login Credentials

The user must provide their Earthdata Login credentials in order to download input data.
If you do not already have an Earthdata account, you can sign up [here](https://urs.earthdata.nasa.gov/home).
Your credentials can be passed to the workflows via environment variables
(`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`) or via your `.netrc` file. If you haven't set up a `.netrc` file
before, check out this [guide](https://harmony.earthdata.nasa.gov/docs#getting-started) to get started.

### Docker Container

The ultimate goal of this project is to create a docker container that can run the workflow within a HyP3
deployment. To run the current version of the project's container, use this command:
```
docker run -it --rm \
    -e EARTHDATA_USERNAME=[YOUR_USERNAME_HERE] \
    -e EARTHDATA_PASSWORD=[YOUR_PASSWORD_HERE] \
    ghcr.io/asfhyp3/hyp3-isce3:latest \
    [WORKFLOW_ARGS]
```

#### Docker Outputs

To retain `hyp3_isce3` output files when running via Docker there are two recommended approaches:

1. Use a volume mount:
   Add the `-w /tmp -v [localdir]:/tmp` flags after docker run. `-w` changes the working directory of the container to `/tmp` and `-v` will mount whichever local directory you choose so that such that hyp3_isce3 outputs are preserved locally.

2. Copy outputs to remote object storage:
   Append the `--bucket` and `--bucket-prefix` to [WORKFLOW_ARGS]. *Only the final output files and zipped archive of those files is uploaded.* This also requires that AWS credentials to write to the bucket are available to the running container. For example, to write outputs to a hypothetical bucket `s3://hypothetical-bucket/test-run/`:
   ```
   docker run -it --rm \
       -e AWS_ACCESS_KEY_ID=[YOUR_KEY] \
       -e AWS_SECRET_ACCESS_KEY=[YOUR_SECRET] \
       -e AWS_SESSION_TOKEN=[YOUR_TOKEN] \
       -e EARTHDATA_USERNAME=[YOUR_USERNAME_HERE] \
       -e EARTHDATA_PASSWORD=[YOUR_PASSWORD_HERE] \
       ghcr.io/asfhyp3/hyp3-isce3:latest \
         [WORKFLOW_ARGS] \
         --bucket "hypothetical-bucket" \
         --bucket-prefix "test-run"
   ```

Tip: you can use [`docker run --env-file`](https://docs.docker.com/reference/cli/docker/container/run/#env) to capture all the necessary environment variables in a single file.

## Developer Setup

1. Ensure that pixi is installed on your system: <https://pixi.sh/latest/installation/>.
2. Clone the `hyp3-isce3` repository and navigate to the root directory of this project
   ```bash
   git clone https://github.com/ASFHyP3/hyp3-isce3.git
   cd hyp3-isce3
   ```
3. setup the development environment
   ```bash
    pixi run install-editable
   ```
4. (optional) [traditional conda-like activation](https://pixi.sh/latest/workspace/environment/#traditional-conda-activate-like-activation) of the pixi environment
   ```bash
   eval "$(pixi shell-hook)"
   ```

    > [!TIP]
    > If you've done (4), you don't need to prefix commands with `pixi run`.
5. (optional) Setup you IDE to work with pixi:
   * PyCharm: https://pixi.sh/dev/integration/editor/jetbrains/
   * VSCode: https://pixi.sh/dev/integration/editor/vscode/

## Background

HyP3 is broken into two components: the cloud architecture/API that manage processing of HyP3 workflows, and Docker container plugins that contain scientific workflows which produce new science products from a variety of data sources (see figure below for the full HyP3 architecture).

The cloud infratstructure-as-code for HyP3 can be found in the main [HyP3 repository](https://github.com/asfhyp3/hyp3). This repository contains a plugin that can be used to process ISCE3-based processing of SAR data.

This project was heavily influenced by the [DockerizedTopsApp](https://github.com/ACCESS-Cloud-Based-InSAR/DockerizedTopsApp) project, which contains a similar workflow that is designed to produce ARIA Sentinel-1 Geocoded Unwrapped Interferogram standard products via HyP3.

## License

The HyP3-ISCE3 plugin is licensed under the Apache License, Version 2 license. See the LICENSE file for more details.

## Code of conduct

We strive to create a welcoming and inclusive community for all contributors to HyP3-ISCE3. As such, all contributors to this project are expected to adhere to our code of conduct.

## Contributing

Contributions to the HyP3-ISCE3 plugin are welcome! If you would like to contribute, please submit a pull request on the GitHub repository.

## Contact Us

Want to talk about HyP3-ISCE3? We would love to hear from you!

Found a bug? Want to request a feature?
[open an issue](https://github.com/ASFHyP3/hyp3-isce3/issues/new)

General questions? Suggestions? Or just want to talk to the team?
[chat with us on gitter](https://gitter.im/ASFHyP3/community)
