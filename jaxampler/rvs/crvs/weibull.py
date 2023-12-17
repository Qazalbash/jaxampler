from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Weibull(ContinuousRV):

    def __init__(self, lmbda: ArrayLike = 1.0, k: ArrayLike = 1.0, name: str = None) -> None:
        self._lmbda = lmbda
        self._k = k
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0.0), "scale must be greater than 0"
        assert jnp.all(self._k > 0.0), "concentration must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.where(
            x < 0, -jnp.inf,
            jnp.log(self._k) - (self._k * jnp.log(self._lmbda)) + (self._k - 1.0) * jnp.log(x) -
            jnp.power(x / self._lmbda, self._k))

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.where(x < 0, 0.0, 1.0 - jnp.exp(-jnp.power(x / self._lmbda, self._k)))

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return self._lmbda * jnp.power(-jnp.log1p(-x), 1.0 / self._k)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        U = jax.random.uniform(key, shape=(N,))
        return self.ppf(U)

    def __repr__(self) -> str:
        string = f"Weibull(lambda={self._lmbda}, k={self._k}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
