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


import sys

import pytest


sys.path.append("../jaxampler")
from jaxampler.rvs import StudentT


class TestStudentT:
    def test_invalid_df(self):
        with pytest.raises(AssertionError):
            StudentT(df=-1.0)

    def test_invalid_scale(self):
        with pytest.raises(AssertionError):
            StudentT(df=1.0, scale=-1.0)

    def test_cdf(self):
        rv = StudentT(df=1.0, loc=-1.0, scale=1.0)
        assert rv.cdf(-1.0) == 0.5
        assert rv.cdf(0.0) == pytest.approx(0.75, abs=1e-4)
