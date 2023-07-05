import numpy as np
import pandas as pd
from autofeat.utils.datasets import list_datasets, get_dataset


def test_list_datasets_fn():
    """
    Test the list_datasets function.
    """
    datasets = list_datasets()
    assert len(datasets) > 0
    assert isinstance(datasets, np.ndarray)
    assert isinstance(datasets[0], str)

    available_datasets = [
        'air passengers',
    ]

    for dataset in available_datasets:
        assert dataset in datasets


def test_get_dataset_fn():
    """
    Test the get_dataset function.
    """
    dataset = get_dataset('air passengers')
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (144, 3)
    assert dataset.columns.tolist() == ['unique_id', 'ds', 'y']
