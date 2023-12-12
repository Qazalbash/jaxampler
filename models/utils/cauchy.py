from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import cauchy as jax_cauchy
from jax.typing import ArrayLike

from .distribution import Distribution


class Cauchy(Distribution):

    def __init__(self, sigma: ArrayLike, loc: ArrayLike = 0, name: str = None) -> None:
        self._sigma = sigma
        self._loc = loc
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0), "sigma must be positive"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logpdf(x, self._loc, self._sigma)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.pdf(x, self._loc, self._sigma)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logcdf(x, self._loc, self._sigma)

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.cdf(x, self._loc, self._sigma)

    @partial(jax.jit, static_argnums=(0,))
    def cdfinv(self, x: ArrayLike) -> ArrayLike:
        return self._loc + self._sigma * jnp.tan(jnp.pi * (x - 0.5))

    def rvs(self, N: int) -> Array:
        return jax.random.cauchy(self.get_key(), self._loc, self._sigma, shape=(N,))

    def __repr__(self) -> str:
        string = f"Cauchy(sigma={self._sigma}, loc={self._loc}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
