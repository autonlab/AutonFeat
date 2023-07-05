#/bin/bash

cd ..
cd ..
# Linting
flake8 autofeat --ignore=E501 --exclude=docs

cd scripts/lint
