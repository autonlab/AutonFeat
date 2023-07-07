#!/bin/bash

# Directory containing the ipynb files
input_directory="../../examples/tutorials"

# Directory to copy the generated files and folders
output_directory="../../docs/tutorials"

# Step 1: Read the list of ipynb files
ipynb_files=$(find "$input_directory" -maxdepth 1 -type f -name "*.ipynb")

# Step 2: Function to convert each file to markdown using nbconvert
convert_to_markdown() {
    local ipynb_file="$1"
    local base_name=$(basename "$ipynb_file")
    local markdown_file="${base_name%.ipynb}.md"
    
    # Convert to markdown
    jupyter nbconvert --execute --to markdown "$ipynb_file" --output-dir="$output_directory"
}

# Iterate over the ipynb files, run the notebooks, perform the conversion and replace doc files
for file in $ipynb_files; do
    # Remove the existing folder
    folder_name_path="${file%'.ipynb'}"_files
    folder_name="$output_directory/${folder_name_path##*/}"
    if [ -d "$folder_name" ]; then
        rm -rf "$folder_name"
    fi
    # Run the notebook and
    # Convert to markdown and copy files
    convert_to_markdown "$file"
done
