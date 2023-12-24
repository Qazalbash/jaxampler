# Copyright 2023 The JAXampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax.typing import ArrayLike

from .binomial import Binomial


class Bernoulli(Binomial):
    """Bernoulli random variable"""

    def __init__(self, p: ArrayLike, name: str = None) -> None:
        """Initialize the Bernoulli random variable.

        Parameters
        ----------
        p : ArrayLike
            Probability of success.
        name : str, optional
            Name of the random variable, by default None
        """
        super().__init__(p, 1, name)

    def __repr__(self) -> str:
        """Return the string representation of the Bernoulli random variable.

        Returns
        -------
        str
            String representation of the Bernoulli random variable.
        """
        string = f"Bernoulli(p={self._p}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
