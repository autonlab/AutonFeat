#/bin/bash

# Generate the diagrams for the documentation
cd ../../examples/visualize_docs

# Run all files
for f in *.py; do
    python $f
done

cd ../../scripts/docs