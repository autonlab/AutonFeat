#!/bin/bash

# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

cd ../..

# Freeze requirements
deactivate
source .venv/bin/activate
pip freeze > requirements.txt
deactivate

# Generate distribution archives and wheel
source .venv_test/bin/activate
python setup.py sdist bdist_wheel

# Docs
cd scripts/docs
bash generate_assets.sh
bash generate_tutorials.sh

cd ../release
