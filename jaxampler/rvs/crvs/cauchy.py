from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.random import KeyArray
from jax.scipy.stats import cauchy as jax_cauchy
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Cauchy(ContinuousRV):

    def __init__(self, sigma: float, loc: float = 0, name: str = None) -> None:
        self._sigma = sigma
        self._loc = loc
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._sigma > 0.0, "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logpdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.pdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logcdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.cdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return self._loc + self._sigma * jnp.tan(jnp.pi * (x - 0.5))

    def rvs(self, N: int = 1, key: KeyArray = None) -> Array:
        if key is None:
            key = self.get_key()
        return jax.random.cauchy(key, self._loc, self._sigma, shape=(N,))

    def __repr__(self) -> str:
        string = f"Cauchy(sigma={self._sigma}, loc={self._loc}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
