from functools import partial

import jax
from jax import Array, jit, lax
from jax import numpy as jnp
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class Triangular(ContinuousRV):

    def __init__(self, low: float = 0, mode: float = 0.5, high: float = 1, name: str = None) -> None:
        self._low = low
        self._mode = mode
        self._high = high
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._low <= self._high), "low must be less than or equal to high"
        assert jnp.all(self._low <= self._mode), "low must be less than or equal to mid"
        assert jnp.all(self._mode <= self._high), "mid must be less than or equal to high"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        conditions = [
            x < self._low,
            (self._low <= x) & (x < self._mode),
            x == self._mode,
            (self._mode < x) & (x <= self._high),
            x > self._high,
        ]
        choices = [
            -jnp.inf,
            jnp.log(2) + jnp.log(x - self._low) - jnp.log(self._high - self._low) - jnp.log(self._mode - self._low),
            jnp.log(2) - jnp.log(self._high - self._low),
            jnp.log(2) + jnp.log(self._high - x) - jnp.log(self._high - self._low) - jnp.log(self._high - self._mode),
            -jnp.inf,
        ]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        conditions = [
            x < self._low,
            (self._low <= x) & (x < self._mode),
            x == self._mode,
            (self._mode < x) & (x < self._high),
            x >= self._high,
        ]
        choices = [
            -jnp.inf,
            2 * jnp.log(x - self._low) - jnp.log(self._high - self._low) - jnp.log(self._mode - self._low),
            jnp.log(0.5),
            jnp.log(1 - ((self._high - x)**2 / ((self._high - self._low) * (self._high - self._mode)))),
            jnp.log(1),
        ]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        _Fc = self.cdf(self._mode)
        ppf_val = jnp.where(
            x < _Fc,
            self._low + lax.sqrt(x * (self._mode - self._low) * (self._high - self._low)),
            self._high - lax.sqrt((1 - x) * (self._high - self._low) * (self._high - self._mode)),
        )
        return ppf_val

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.triangular(key, left=self._low, right=self._high, mode=self._mode, shape=(N,))

    def __repr__(self) -> str:
        string = f"Triangular(low={self._low}, mode={self._mode}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
