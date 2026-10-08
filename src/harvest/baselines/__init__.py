"""Adapters for scientifically qualified HARVEST baseline comparisons.

Concrete adapters are intentionally not imported here.  Keeping package
initialization light avoids circular imports with :mod:`harvest.routing`.
"""

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig

__all__ = ["BaselineAdapter", "BaselineResult", "BaselineRunConfig"]
