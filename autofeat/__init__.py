# Package imports
from .core import (
    SlidingWindow, Transform, Preprocess,
)

from .common import (
    MeanTransform, MaxTransform, MinTransform,
    QuantileTransform, RangeTransform, IQRTransform,
    MedianTransform, StdTransform, VarTransform,
    NValidTransform, DataDensityTransform, DataSparsityTransform,
    SkewnessTransform, KurtosisTransform, EntropyTransform,
    CrossEntropyTransform, SampleEntropyTransform,
)

import autofeat.functional as functional
import autofeat.preprocess as preprocess
import autofeat.utils as utils

# For linter
__all__ = [
    'SlidingWindow',
    'Transform',
    'Preprocess',

    'MeanTransform',
    'MaxTransform',
    'MinTransform',
    'QuantileTransform',
    'RangeTransform',
    'IQRTransform',
    'MedianTransform',
    'StdTransform',
    'VarTransform',
    'NValidTransform',
    'DataDensityTransform',
    'DataSparsityTransform',
    'SkewnessTransform',
    'KurtosisTransform',
    'EntropyTransform',
    'CrossEntropyTransform',
    'SampleEntropyTransform',

    'functional',
    'preprocess',
    'utils',
]

# Property imports
import os
from typing import List


class SetupProperties(object):
    """
    A class to store all the properties needed to setup the package.
    """
    def __init__(self) -> None:
        """
        Properties needed to setup the package.
        """
        self.name = "autofeat"
        self.version = '0.1.0'
        self.description = self._read_file('README.md')
        self.author = ['Dhruv Srikanth', 'Auton Lab']
        self.author_email = 'dsrikant@andrew.cmu.edu'
        self.license = self._read_file('LICENSE')
        self.url = 'https://autonlab.github.io/AutoFeat'
        with open('requirements.txt') as f:
            self.packages = [requirement.rstrip() for requirement in f]

    def _read_file(self, file_path: str) -> str:
        """
        Read a file and return the contents of the file.

        Args:
            file_path (str): The path to the file.

        Returns:
            The contents of the file.
        """
        with open(file_path) as f:
            return f.read()

    def get_name(self) -> str:
        """
        Get the name of the package.

        Returns:
            The name of the package.
        """
        return self.name

    def get_version(self) -> str:
        '''
        Get the version of the package.

        Returns:
            The version of the package.
        '''
        return self.version

    def get_description(self) -> str:
        """
        Get the description of the package which is read from the README.md file.

        Returns:
            The description of the package.
        """
        return self.description

    def get_author(self) -> List[str]:
        """
        Get the author of the package.

        Returns:
            The author of the package.
        """
        return self.author

    def get_author_email(self) -> str:
        """
        Get the author email of the package.

        Returns:
            The author email of the package.
        """
        return self.author_email

    def get_url(self) -> str:
        """
        Get the url of the package.

        Returns:
            The url of the package.
        """
        return self.url

    def get_packages(self) -> List[str]:
        """
        Get the packages used by the package.

        Returns:
            The packages used by the package.
        """
        return self.packages

    def get_license(self) -> str:
        """
        Get the license of the package which is read from the LICENSE file.

        Returns:
            The license of the package.
        """
        return self.license
