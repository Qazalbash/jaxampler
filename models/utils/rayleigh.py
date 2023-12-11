from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .distribution import Distribution


class Rayleigh(Distribution):

    def __init__(self, sigma: ArrayLike, name: str = None) -> None:
        self._sigma = sigma
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0), "sigma must be positive"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        logpdf_val = jnp.log(x) - 0.5 * jnp.power(x / self._sigma, 2) - 2 * jnp.log(self._sigma)
        return jnp.where(x >= 0, logpdf_val, -jnp.inf)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.log1p(-jnp.exp(-0.5 * jnp.power(x / self._sigma, 2)))
        return jnp.where(x >= 0, logcdf_val, -jnp.inf)

    @partial(jax.jit, static_argnums=(0,))
    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        logcdfinv_val = jnp.log(self._sigma) + 0.5 * jnp.log(-2 * jnp.log1p(-x))
        return jnp.where(x >= 0, logcdfinv_val, -jnp.inf)

    def logrvs(self, N: int) -> Array:
        U = jax.random.uniform(jax.random.PRNGKey(0), shape=(N,))
        return jnp.log(self._sigma) + 0.5 * jnp.log(-2 * jnp.log(U))

    def rvs(self, N: int = 1) -> Array:
        return jax.random.rayleigh(jax.random.PRNGKey(0), scale=self._sigma, shape=(N,))

    def __repr__(self) -> str:
        string = f"Rayleigh(sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
