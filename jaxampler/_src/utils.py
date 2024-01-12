#  Copyright 2023 The Jaxampler Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing_extensions import Unpack

import numpy as np
from jax import lax, numpy as jnp
from jax._src import core
from jaxtyping import Integer

from jaxampler._src.typing import Numeric


def jx_cast(
    *args: Numeric,
) -> tuple[tuple[Any, ...], Unpack[tuple[Any, ...]]]:
    """Cast provided arguments to `jnp.array` and checks if they can be
    broadcast.

    Parameters
    ----------
    *args:
        Arguments to cast and check.

    Returns
    -------
    list[Array]
        List of cast arguments.
    """
    # partially taken from the implementation of `jnp.broadcast_arrays`
    shapes = [np.shape(arg) for arg in args]
    if not shapes or all(core.definitely_equal_shape(shapes[0], s) for s in shapes):
        result_shape = shapes[0]
    else:
        result_shape: tuple[int, ...] = lax.broadcast_shapes(*shapes)
    return result_shape, *tuple(jnp.asarray(arg) for arg in args)


fact = [1, 1, 2, 6, 24, 120, 720, 5_040, 40_320, 362_880, 3_628_800]


def nPr(n: Integer, r: Integer) -> Integer:
    """Calculates the number of permutations of `r` objects out of `n`

    Parameters
    ----------
    n : Integer
        total objects
    r : Integer
        selected objects

    Returns
    -------
    Integer
        number of permutations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return fact[n] // fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] // fact[n - r]


def nCr(n: Integer, r: Integer) -> Integer:
    """Calculates the number of combinations of `r` objects out of `n`

    Parameters
    ----------
    n : Integer
        total objects
    r : Integer
        selected objects

    Returns
    -------
    Integer
        number of combinations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return (fact[n] // fact[r]) // fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] // (fact[r] * fact[n - r])
