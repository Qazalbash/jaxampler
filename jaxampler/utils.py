# Copyright 2023 The Jaxampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import jax
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike


def jx_cast(*args: ArrayLike) -> list[Array]:
    """Cast provided arguments to jnp.array and broadcasts them
    with respect to each other

    Parameters
    ----------
    *args:
        Arguments to cast and broadcast

    Returns
    -------
    list[Array]
        broadcasted arrays
    """
    return jnp.broadcast_arrays(*[jnp.asarray(a) for a in args])


fact = [1, 1, 2, 6, 24, 120, 720, 5_040, 40_320, 362_880, 3_628_800]


def nPr(n: int, r: int) -> int:
    """Calculates the number of permutations of `r` objects out of `n`

    Parameters
    ----------
    n : int
        total objects
    r : int
        selected objects

    Returns
    -------
    int
        number of permutations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return fact[n] / fact[r] / fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] / fact[n - r]


def nCr(n: int, r: int) -> int:
    """Calculates the number of combinations of `r` objects out of `n`

    Parameters
    ----------
    n : int
        total objects
    r : int
        selected objects

    Returns
    -------
    int
        number of combinations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return fact[n] / fact[r] / fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] / fact[r] / fact[n - r]


def new_prn_key(key: Array = None) -> Array:
    """Get a new JAX random key.

    This function is used to generate a new JAX random key if
    the user does not provide one. The key is generated using
    the JAX random.PRNGKey function. The key is split into
    two keys, the first of which is returned. The second key
    is discarded.

    Parameters
    ----------
    key : Array, optional
        JAX random key, by default None

    Returns
    -------
    Array
        New JAX random key.
    """
    if key is None:
        new_key = jax.random.PRNGKey(random.randint(0, 1000_000))
    else:
        new_key, _ = jax.random.split(key)
    return new_key
