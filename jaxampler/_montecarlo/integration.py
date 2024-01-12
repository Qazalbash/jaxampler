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

from abc import abstractmethod
from typing import Optional

from .._jobj import JObj
from .._typing import Numeric


class Integration(JObj):
    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name

    @abstractmethod
    def check_params(self, *args, **kwargs) -> None:
        raise NotImplementedError("Function should be implemented in the sub-class.")

    @abstractmethod
    def compute_integral(self, *args, **kwargs) -> Numeric:
        raise NotImplementedError("Function should be implemented in the sub-class.")

    def __repr__(self) -> str:
        return f"Integration(name={self._name})"
