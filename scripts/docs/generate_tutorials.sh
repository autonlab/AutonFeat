#!/bin/bash

# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    jupyter nbconvert --to markdown "$ipynb_file" --output-dir="$output_directory"
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
