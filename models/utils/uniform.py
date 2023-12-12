from functools import partial

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.scipy.stats import uniform as jax_uniform
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Uniform(ContinuousRV):

    def __init__(self, low: ArrayLike = 0.0, high: ArrayLike = 1.0, name: str = None) -> None:
        self._low = low
        self._high = high
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "All low must be less than high"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_uniform.logpdf(x, loc=self._low, scale=self._high - self._low)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.where((self._low <= x) & (x <= self._high),
                               lax.log(x - self._low) - lax.log(self._high - self._low), -jnp.inf)
        logcdf_val = jnp.where(x <= self._high, logcdf_val, jnp.log(1.0))

        return logcdf_val

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_uniform.pdf(x, loc=self._low, scale=self._high - self._low)

    def rvs(self, N: int) -> Array:
        return jax.random.uniform(self.get_key(), minval=self._low, maxval=self._high, shape=(N,), dtype=jnp.float32)

    def __repr__(self) -> str:
        string = f"Uniform(low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
