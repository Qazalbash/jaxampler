from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import pareto as jax_pareto
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Pareto(ContinuousRV):

    def __init__(self, alpha: float, scale: float, name: str = None) -> None:
        self._alpha = alpha
        self._scale = scale
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._alpha > 0.0, "alpha must be greater than 0"
        assert self._scale > 0.0, "scale must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_pareto.logpdf(x, self._alpha, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_pareto.pdf(x, self._alpha, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.where(self._scale <= x, jnp.log1p(-jnp.power(self._scale / x, self._alpha)), -jnp.inf)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        conditions = [
            x < 0.0,
            (0.0 <= x) & (x < 1.0),
            1.0 <= x,
        ]
        choices = [
            -jnp.inf,
            jnp.log(self._scale) - (1.0 / self._alpha) * jnp.log1p(-x),
            jnp.log(1.0),
        ]
        return jnp.select(conditions, choices)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        return jax.random.pareto(key, self._alpha, shape=(N,)) * self._scale

    def __repr__(self) -> str:
        string = f"Pareto(alpha={self._alpha}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
