from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import pareto as jax_pareto
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Pareto(ContinuousRV):

    def __init__(self, alpha: ArrayLike, scale: ArrayLike, name: str = None) -> None:
        self._alpha = alpha
        self._scale = scale
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0.0), "alpha must be greater than 0"
        assert jnp.all(self._scale > 0.0), "scale must be greater than 0"

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
        logcdfinv_val = jnp.log(self._scale) - (1.0 / self._alpha) * jnp.log(1 - x)
        logcdfinv_val = jnp.where(0.0 <= x, logcdfinv_val, -jnp.inf)
        logcdfinv_val = jnp.where(x < 1.0, logcdfinv_val, jnp.log(1.0))
        return logcdfinv_val

    def rvs(self, N: int = 1) -> Array:
        return jax.random.pareto(self.get_key(), self._alpha, shape=(N,)) * self._scale

    def __repr__(self) -> str:
        string = f"Pareto(alpha={self._alpha}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
