from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import expon as jax_expon
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Exponential(ContinuousRV):

    def __init__(self, lmbda: ArrayLike, name: str = None) -> None:
        self._lmbda = lmbda
        self.check_params()
        self._scale = 1.0 / lmbda
        self._logZ = self.logZ()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0), "lamda must be positive"

    @partial(jax.jit, static_argnums=(0,))
    def logZ(self) -> ArrayLike:
        return -jnp.log(self._lmbda)

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_expon.logpdf(x, scale=self._scale)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_expon.pdf(x, scale=self._scale)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.log1p(-jnp.exp(-self._lmbda * x))
        return jnp.where(x >= 0, logcdf_val, -jnp.inf)

    @partial(jax.jit, static_argnums=(0,))
    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        logcdfinv_val = jnp.log(-jnp.log1p(-x)) + self._logZ
        return jnp.where(x >= 0, logcdfinv_val, -jnp.inf)

    def logrvs(self, N: int) -> Array:
        U = jax.random.uniform(self.get_key(), shape=(N,))
        return jnp.log(-jnp.log(U)) + self._logZ

    def __repr__(self) -> str:
        string = f"Exponential(lamda={self._lmbda}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
