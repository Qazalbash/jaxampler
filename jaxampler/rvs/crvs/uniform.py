from functools import partial

import jax
from jax import Array, jit, lax
from jax import numpy as jnp
from jax.scipy.stats import uniform as jax_uniform
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Uniform(ContinuousRV):

    def __init__(self, low: float = 0.0, high: float = 1.0, name: str = None) -> None:
        self._low = low
        self._high = high
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._low < self._high, "All low must be less than high"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_uniform.logpdf(x, loc=self._low, scale=self._high - self._low)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        conditions = [x < self._low, (self._low <= x) & (x <= self._high), self._high < x]
        choice = [
            -jnp.inf,
            lax.log(x - self._low) - lax.log(self._high - self._low),
            jnp.log(1.0),
        ]
        return jnp.select(conditions, choice)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_uniform.pdf(x, loc=self._low, scale=self._high - self._low)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        return jax.random.uniform(key, minval=self._low, maxval=self._high, shape=(N,))

    def __repr__(self) -> str:
        string = f"Uniform(low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
