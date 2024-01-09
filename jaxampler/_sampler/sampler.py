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

from jax import Array

from .._rvs import ContinuousRV, GenericRV
from ..jobj import JObj


class Sampler(JObj):
    """Sampler is a base class for all samplers."""

    def __init__(self, name: str = None) -> None:
        """Initializes a Sampler object."""
        super().__init__(name)

    def check_rv(self, rv: GenericRV) -> None:
        """Checks if the given random variable is a valid random variable for the sampler.

        If the random variable is not valid, an AssertionError is raised.

        Parameters
        ----------
        rv : GenericRV
            The random variable to check.
        """
        assert isinstance(rv, GenericRV), f"rv must be a GenericRV object, got {rv}"
        assert isinstance(rv, ContinuousRV), f"rv must be a ContinuousRV object, got {rv}"

    @abstractmethod
    def sample(self, *args, **kwargs) -> Array:
        """Samples from the given random variable.

        It runs the sampling algorithm and returns the samples.

        Parameters
        ----------
        rv : GenericRV
            The random variable to sample from.
        N : int, optional
            Number of samples, by default 1
        key : Array, optional
            The key to use for sampling, by default None

        Returns
        -------
        Array
            The samples.

        Raises
        ------
        NotImplementedError
            This method must be implemented by the subclass.
        """
        raise NotImplementedError
