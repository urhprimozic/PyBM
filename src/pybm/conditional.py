# handles if-else statemtets with constants inside conditions in the ode, by relaxing the discontinuity to a smooth transition
from typing import Literal
import torch
import numpy as np
import scipy


class RelaxedIfElse:
    """
    Smooth  if/elif/.../else. 

    

    Pogoji se ne podajajo kot bool, ampak kot 'signed distance':
        diff > 0  -> pogoj je (mehko) resničen
        diff < 0  -> pogoj je (mehko) neresničen
        diff = 0  -> na meji (utež = 0.5)

    Primer: "if t <= c" postane diff = c - t
    """

    def __init__(self, eps=1, method="tanh", engine:Literal["torch", "scipy", "jax"]= "scipy"):
        self.eps = eps
        self.method = method
        self._branches = []   # [(diff, value), ...] v vrstnem redu kot elif verige
        self._default = None
        self.engine = engine

    def _weight(self, diff):
        diff = np.asarray(diff, dtype=float)
        if self.method == "tanh":
            if self.engine == "torch":
                diff = torch.as_tensor(diff, dtype=torch.float32)
                return 0.5 * (1 + torch.tanh(diff / self.eps))
            elif self.engine == "scipy":
                return 0.5 * (1 + np.tanh(diff / self.eps))
            else:
                raise NotImplementedError(f"Engine {self.engine} is not implemented for method {self.method}.")
        elif self.method == "erf":
            if self.engine == "torch":
                diff = torch.as_tensor(diff, dtype=torch.float32)
                return 0.5 * (1 + torch.erf(diff / (self.eps * torch.sqrt(2))))
            elif self.engine == "scipy":
                return 0.5 * (1 + scipy.special.erf(diff / (self.eps * np.sqrt(2))))
            elif self.engine == "jax":
                raise NotImplementedError("JAX engine is not implemented yet.")
            else:
                raise ValueError(f"Unknown engine: {self.engine}")
        elif self.method == "linear":
            if self.engine == "torch":
                diff = torch.as_tensor(diff, dtype=torch.float32)
                return torch.clip((diff / self.eps + 1) / 2, 0.0, 1.0)
            elif self.engine == "scipy":
                return np.clip((diff / self.eps + 1) / 2, 0.0, 1.0)
            else:
                raise NotImplementedError(f"Engine {self.engine} is not implemented for method {self.method}.")
        else:
            raise ValueError(f"neznana metoda: {self.method}")

    # ---- fluent API: If(...).Elif(...).Elif(...).Else(...) ----
    def If(self, diff, value):
        self._branches.append((diff, value))
        return self

    def Elif(self, diff, value):
        self._branches.append((diff, value))
        return self

    def Else(self, value):
        self._default = value
        return self._build()

    def _build(self):
        # vsaka veja "porabi" samo tisto verjetnostno maso, ki je
        # prejšnje veje še niso zajele -> pravi analog if/elif/else
        remaining = 1.0
        result = 0.0
        for diff, value in self._branches:
            w = self._weight(diff) * remaining
            result = result + w * value
            remaining = remaining - w
        if self._default is not None:
            result = result + remaining * self._default
        return result
    