from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import gamma as jax_gamma
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Gamma(ContinuousRV):

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, name: str = None) -> None:
        self._alpha = alpha
        self._beta = beta
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0), "All alpha must be greater than 0"
        assert jnp.all(self._beta > 0), "All beta must be greater than 0"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_gamma.logpdf(x, self._alpha, scale=1 / self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_gamma.cdf(x, self._alpha, scale=1 / self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def rvs(self, N: int) -> Array:
        return jax.random.gamma(self.get_key(), self._alpha, shape=(N,)) / self._beta

    def __repr__(self) -> str:
        string = f"Gamma(alpha={self._alpha}, beta={self._beta}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
