#/bin/bash

cd ..
cd ..
# Linting
flake8 autonfeat --ignore=E501 --exclude=docs

cd scripts/lint
