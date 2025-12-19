"""ColliderML: A modern machine learning library for high-energy physics data analysis."""

__version__ = "0.3.0"

from . import core
from . import physics
from . import polars
from . import utils

__all__ = ["core", "physics", "polars", "utils"]