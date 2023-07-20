from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

user_requirements = [
    requirement.strip()
    for requirement in open(
        path.join(here, 'requirements.txt')
    ).readlines()
]

setup(
    name='autonfeat',
    version='0.1.0',
    author=[
        'Dhruv Srikanth',
        'Auton Lab'
    ],
    author_email='dsrikant@andrew.cmu.edu',
    description='A High Performance Library for Time-Series Featurization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://autonlab.github.io/AutoFeat',
    keywords=[
        'artificial intelligence',
        'machine learning',
        'data science',
        'data analysis',
        'time-series',
        'featurization',
        'forecasting',
        'high-performance computing',
    ],
    packages=find_packages(where='autonfeat'),
    package_dir={'': 'autonfeat'},
    package_data={'': ['autonfeat/utils/datasets/data/*.csv']},
    install_requires=user_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
