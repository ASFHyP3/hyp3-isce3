"""Shared numeric constants."""

# Degrees padded around the AOI (or frame) when searching tiled ancillary data, so the
# staged mosaic covers the crop margin and radar-processing edges.
ANCILLARY_BUFFER_DEG = 1.0

# Degrees padded around the AOI when staging the DEM for a subset run (~11 km), past the
# 512-px crop margin's ground extent (~5-6 km).
DEM_SUBSET_BUFFER_DEG = 0.1
