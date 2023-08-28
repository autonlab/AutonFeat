#!/bin/bash

# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

cd ../..
source .venv_test/bin/activate

# Lint
cd scripts/lint
bash lint.sh

# Test
cd ../test
bash test.sh

cd ../release


