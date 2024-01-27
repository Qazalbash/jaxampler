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

from __future__ import annotations

import random
from typing import Optional

import jax
from jaxtyping import Array


class JObj(object):
    """Jaxampler generic object class"""

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name

    @property
    def name(self):
        return self._name

    @staticmethod
    def get_key(key: Optional[Array] = None) -> Array:
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

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        raise NotImplementedError
