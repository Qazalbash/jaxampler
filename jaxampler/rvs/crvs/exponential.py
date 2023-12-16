from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import expon as jax_expon
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Exponential(ContinuousRV):

    def __init__(self, lmbda: float, name: str = None) -> None:
        self._lmbda = lmbda
        self.check_params()
        self._scale = 1.0 / lmbda
        self._logZ = self.logZ()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._lmbda > 0.0, "lmbda must be positive"

    @partial(jit, static_argnums=(0,))
    def logZ(self) -> ArrayLike:
        return -jnp.log(self._lmbda)

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_expon.logpdf(x, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_expon.pdf(x, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.log1p(-jnp.exp(-self._lmbda * x))
        return jnp.where(x >= 0, logcdf_val, -jnp.inf)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        logcdfinv_val = jnp.log(-jnp.log1p(-x)) + self._logZ
        return jnp.where(x >= 0, logcdfinv_val, -jnp.inf)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        U = jax.random.uniform(key, shape=(N,))
        rvs_val = jnp.log(-jnp.log(U)) + self._logZ
        return jnp.exp(rvs_val)

    def __repr__(self) -> str:
        string = f"Exponential(lamda={self._lmbda}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
