# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
from autonfeat.utils.datasets import list_datasets, get_dataset


def test_list_datasets_fn():
    """
    Test the list_datasets function.
    """
    datasets = list_datasets()
    assert len(datasets) > 0
    assert isinstance(datasets, np.ndarray)
    assert isinstance(datasets[0], str)

    available_datasets = [
        'airline passengers',
    ]

    for dataset in available_datasets:
        assert dataset in datasets


def test_get_dataset_fn():
    """
    Test the get_dataset function.
    """
    dataset = get_dataset('airline passengers')
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (144, 3)
    assert dataset.columns.tolist() == ['uid', 'datestamp', 'passengers']
