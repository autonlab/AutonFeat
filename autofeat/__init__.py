# Package imports
from .core import (
    SlidingWindow, Transform, Preprocess,
)

from .common import (
    MeanTransform, MaxTransform, MinTransform,
    QuantileTransform, RangeTransform, IQRTransform,
    MedianTransform, StdTransform, VarTransform,
    NValidTransform, DataDensityTransform, DataSparsityTransform,
)

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
]

# Property imports
import os


class SetupProperties(object):
    """
    A class to store all the properties needed to setup the package.
    """
    def __init__(self):
        """
        Properties needed to setup the package.
        """
        self.name = "autofeat"
        self.version = '0.0.1'
        self.description = os.read('README.md')
        self.author = 'Dhruv Srikanth'
        self.author_email = 'dsrikant@andrew.cmu.edu'
        self.url = None
        self.packages = [
            'distutils',
            'distutils.command',
            'numpy',
            'typing',
            'numba',
        ]

    def get_name(self):
        """
        Get the name of the package.

        Returns:
            The name of the package.
        """
        return self.name

    def get_version(self):
        '''
        Get the version of the package.

        Returns:
            The version of the package.
        '''
        return self.version

    def get_description(self):
        """
        Get the description of the package which is read from the README.md file.

        Returns:
            The description of the package.
        """
        return self.description

    def get_author(self):
        """
        Get the author of the package.

        Returns:
            The author of the package.
        """
        return self.author

    def get_author_email(self):
        """
        Get the author email of the package.

        Returns:
            The author email of the package.
        """
        return self.author_email

    def get_url(self):
        """
        Get the url of the package.

        Returns:
            The url of the package.
        """
        return self.url

    def get_packages(self):
        """
        Get the packages used by the package.

        Returns:
            The packages used by the package.
        """
        return self.packages
