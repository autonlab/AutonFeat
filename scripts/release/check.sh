#/bin/bash

cd ../..
source .venv_test/bin/activate

# Lint
cd scripts/lint
bash lint.sh

# Test
cd ../test
bash test.sh

cd ../release


