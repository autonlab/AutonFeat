import importlib
import numpy as np
import pandas as pd


def get_dataset_map() -> dict:
    """
    Get a map of all available datasets.

    Returns:
        A map of all available datasets.
    """
    dataset_path = importlib.util.find_spec('autofeat.utils.datasets').origin.replace('__init__.py', 'data')
    available_datasets = {
        'air passengers': f'{dataset_path}/air_passengers.csv',
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
