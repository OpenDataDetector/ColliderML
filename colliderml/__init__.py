"""ColliderML: A modern machine learning library for high-energy physics data analysis."""

__version__ = "0.3.1"

from . import clustering
from . import core
from . import physics
from . import polars
from . import utils

__all__ = ["clustering", "core", "physics", "polars", "utils"]