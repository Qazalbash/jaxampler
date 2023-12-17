from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Rayleigh(ContinuousRV):

    def __init__(self, sigma: float, name: str = None) -> None:
        self._sigma = sigma
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._sigma > 0.0, "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        logpdf_val = jnp.log(x) - 0.5 * jnp.power(x / self._sigma, 2) - 2 * jnp.log(self._sigma)
        return jnp.where(x >= 0, logpdf_val, -jnp.inf)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.log1p(-jnp.exp(-0.5 * jnp.power(x / self._sigma, 2)))
        return jnp.where(x >= 0, logcdf_val, -jnp.inf)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        logppf_val = jnp.log(self._sigma) + 0.5 * jnp.log(-2 * jnp.log1p(-x))
        return jnp.where(x >= 0, logppf_val, -jnp.inf)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.rayleigh(key, scale=self._sigma, shape=(N,))

    def __repr__(self) -> str:
        string = f"Rayleigh(sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
