"""
Interpolations for exogenous data, in addition to `pybm.model.LinearInterpolation` and
`pybm.model.GaussianProcessInterpolation`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pybm.model import TimeSeries


def _numpy(a) -> np.ndarray:
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


class StepInterpolation(TimeSeries):
    """
    Piecewise constant interpolation ("zero-order hold"): sample i holds on [t_i, t_{i+1}). Before
    t_0 the first value holds, from the last sample on the last value.

    For per-interval data: daily totals and means, per-day indices, categorical codes. Inside an
    interval nothing is known beyond the one value, and values of neighbouring intervals don't mix,
    so the integral over each interval is exactly its sample times its length, and a category is
    never between two codes.
    """

    def __call__(self, t, *args: Any, **kwds: Any) -> Any:
        if torch.is_tensor(t):
            dtype = t.dtype if t.is_floating_point() else torch.float64
            grid = torch.as_tensor(self.t, dtype=dtype, device=t.device)
            values = torch.as_tensor(self.x, dtype=dtype, device=t.device)
            index = torch.searchsorted(grid, t.to(dtype).reshape(-1).contiguous(), right=True) - 1
            return values[index.clamp(0, len(grid) - 1)].reshape(t.shape)
        if type(t).__module__.startswith("jax"):
            import jax.numpy as jnp

            grid = jnp.asarray(_numpy(self.t))
            index = jnp.clip(jnp.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 1)
            return jnp.asarray(_numpy(self.x), dtype=float)[index]

        grid = _numpy(self.t)
        index = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 1)
        x = _numpy(self.x).astype(float)[index]
        return float(x) if np.ndim(x) == 0 else x
