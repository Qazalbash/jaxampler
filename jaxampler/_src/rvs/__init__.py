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

from .bernoulli import Bernoulli as Bernoulli
from .beta import Beta as Beta
from .binomial import Binomial as Binomial
from .boltzmann import Boltzmann as Boltzmann
from .cauchy import Cauchy as Cauchy
from .chi2 import Chi2 as Chi2
from .crvs import ContinuousRV as ContinuousRV
from .drvs import DiscreteRV as DiscreteRV
from .exponential import Exponential as Exponential
from .gamma import Gamma as Gamma
from .geometric import Geometric as Geometric
from .logistic import Logistic as Logistic
from .lognormal import LogNormal as LogNormal
from .normal import Normal as Normal
from .pareto import Pareto as Pareto
from .poisson import Poisson as Poisson
from .rayleigh import Rayleigh as Rayleigh
from .rvs import GenericRV as GenericRV
from .studentt import StudentT as StudentT
from .triangular import Triangular as Triangular
from .truncnormal import TruncNormal as TruncNormal
from .truncpowerlaw import TruncPowerLaw as TruncPowerLaw
from .uniform import Uniform as Uniform
from .weibull import Weibull as Weibull
