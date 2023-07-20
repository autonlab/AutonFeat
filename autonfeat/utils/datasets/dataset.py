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

import importlib
import numpy as np
import pandas as pd


def get_dataset_map() -> dict:
    """
    Get a map of all available datasets.

    Returns:
        A map of all available datasets.
    """
    dataset_path = importlib.util.find_spec('autonfeat.utils.datasets').origin.replace('__init__.py', 'data')
    available_datasets = {
        'airline passengers': f'{dataset_path}/airline_passengers.csv',
    }
    return available_datasets


def list_datasets() -> np.ndarray:
    """
    List all available datasets.

    Returns:
        A list of all available datasets.
    """
    return np.array(list(get_dataset_map().keys()))


def get_dataset(name: str) -> pd.DataFrame:
    """
    Get a dataset by name.

    Args:
        name: The name of the dataset.

    Returns:
        The dataset.

    Raises:
        ValueError: If the dataset is not found.
    """
    available_datasets = list_datasets()
    if name not in available_datasets:
        raise ValueError(f'Dataset {name} not found. Available datasets: {available_datasets}')

    dataset_map = get_dataset_map()
    dataset_path = dataset_map[name]
    return pd.read_csv(dataset_path)
