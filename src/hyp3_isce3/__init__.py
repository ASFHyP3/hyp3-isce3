"""HyP3 plugin for isce3 processing."""

from importlib.metadata import version

import numpy as np


# RAiDER 0.5.3 integrates the zenith delay with np.trapz, which numpy 2.4 removed; np.trapezoid is the same function.
# Set here so it is in place in every process that imports the package, including isce3_fast's spawned workers.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

__version__ = version(__name__)

__all__ = [
    '__version__',
]
