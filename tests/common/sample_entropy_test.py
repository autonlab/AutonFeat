# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import SampleEntropyTransform
import pytest


def test_sample_entropy():
    """
    Test sample entropy transform.
    """

    x = np.random.rand(100)

    tf = SampleEntropyTransform()
    y_hat = tf(x, m=2, r=0.2)

    y = 0.0

    # assert pytest.approx(y_hat) == y
