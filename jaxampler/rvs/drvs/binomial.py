from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import binom as jax_binom
from jax.typing import ArrayLike

from .drvs import DiscreteRV


class Binomial(DiscreteRV):

    def __init__(self, p: float, n: int, name: str = None) -> None:
        self._p = p
        self._n = n
        self.check_params()
        self._q = 1.0 - p
        super().__init__(name)

    def check_params(self) -> None:
        assert (self._p >= 0.0) and (self._p <= 1.0), "p must be in [0, 1]"
        assert type(self._n) == int, "n must be an integer"
        assert self._n > 0, "n must be positive"

    @partial(jit, static_argnums=(0))
    def logpmf(self, k: ArrayLike) -> ArrayLike:
        return jax_binom.logpmf(k, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def pmf(self, k: ArrayLike) -> ArrayLike:
        return jax_binom.pmf(k, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, k: ArrayLike) -> ArrayLike:
        return jnp.log(self.cdf(k))

    @partial(jit, static_argnums=(0,))
    def cdf(self, k: ArrayLike) -> ArrayLike:
        x = jnp.arange(0, self._n + 1, dtype=jnp.int32)
        complete_cdf = jnp.cumsum(self.pmf(x))
        cond = [k < 0, k >= self._n, jnp.logical_and(k >= 0, k < self._n)]
        return jnp.select(cond, [0.0, 1.0, complete_cdf[k]])

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        return jax.random.binomial(key=key, n=self._n, p=self._p, shape=(N,))

    def __repr__(self) -> str:
        string = f"Binomial(p={self._p}, n={self._n}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
