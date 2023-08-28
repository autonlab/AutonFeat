#!/bin/bash

# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

# Generate the diagrams for the documentation
cd ../../examples/visualize_docs

# Run all files
for f in *.py; do
    python $f
done

cd ../../scripts/docs