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

from jax import Array

from .utils import new_prn_key


class JObj(object):
    """Jaxampler generic object class"""

    def __init__(self, name: str = None) -> None:
        self._name = name

    @staticmethod
    def get_key(key: Array = None) -> Array:
        return new_prn_key(key)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        raise NotImplementedError
