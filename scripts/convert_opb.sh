#!/bin/bash

# Directory containing the JSON files
INPUT_DIR="../data/anon_jsons"
OUTPUT_DIR="../data/opb_instances"
PYTHON_SCRIPT="../scripts/print_opb.py"
DO_SYMMBREAK="true"

# Modify output directory based on symmetry breaking flag
if [ "$DO_SYMMBREAK" == "true" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_symmbreak"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"


# Loop over all .json files in the directory
for file in "$INPUT_DIR"/*.json; do

    # Check if the file actually exists (to avoid issues if no .json files)
    [ -e "$file" ] || continue

    # Extract the base name without extension
    base_name=$(basename "$file" .json)

    # Define the output file with .opb extension
    output_file="$OUTPUT_DIR/$base_name.opb"

    echo "Processing file $file -> $output_file"

    # Call the Python script and capture its output
    result=$(python3 "$PYTHON_SCRIPT" "$file" "$DO_SYMMBREAK")

    # Write the output to the new file
    echo "$result" > "$output_file"

done
