FROM ghcr.io/prefix-dev/pixi:latest

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="HyP3 isce3 "
LABEL org.opencontainers.image.description="HyP3 plugin for isce3 processing."
LABEL org.opencontainers.image.vendor="Alaska Satellite Facility"
LABEL org.opencontainers.image.authors="username_for_github_actions <email_for@github_actions.com>"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://github.com/ASFHyP3/hyp3-isce3"
LABEL org.opencontainers.image.source="https://github.com/ASFHyP3/hyp3-isce3"
LABEL org.opencontainers.image.documentation="https://hyp3-docs.asf.alaska.edu"

# Dynamic lables to define at build time via `docker build --label`
# LABEL org.opencontainers.image.created=""
# LABEL org.opencontainers.image.version=""
# LABEL org.opencontainers.image.revision=""

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends git g++ unzip vim patch wget ca-certificates && \
  apt-get clean && rm -rf /var/lib/apt/lists/*


USER 1000
SHELL ["/bin/bash", "-l", "-c"]

#USER root

WORKDIR /hyp3-isce3/

#RUN chown -R 1000:1000 .

#USER 1000
COPY --chown=1000:1000 pyproject.toml pixi.lock ./


RUN pixi install --locked && \
  pixi shell-hook -s bash >> /home/ubuntu/.profile

# Install each dep on a separate layer so we cache the 10 minute long process
RUN pixi run install-isce3

COPY --chown=1000:1000 . .

RUN pixi run install-editable


WORKDIR /home/ubuntu/

ENTRYPOINT ["/hyp3-isce3/src/hyp3_isce3/etc/entrypoint.sh"]
CMD ["-h"]
