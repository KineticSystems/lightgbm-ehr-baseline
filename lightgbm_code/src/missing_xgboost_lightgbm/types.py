"""Small shared data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray | sparse.csr_matrix
    names: tuple[str, ...]
