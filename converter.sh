#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR INCLUDE_DIR"
    exit 1
fi

# Define the input, output, and include directories from the command-line arguments
INPUT_DIR=$1
OUTPUT_DIR=$2
INCLUDE_DIR=$3

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Define the tool command
TOOL_COMMAND="CUDAIntegratedTransformerTool"

# Define additional arguments for the tool
TOOL_ARGS="--cuda-gpu-arch=sm_86 -I$INCLUDE_DIR"

# List of boolean flags, including the new flag
FLAGS=("convert-double-to-float" "change-Kernel" "dim3" "change-specific"
       "remove_synch_thread_to_null" "remove_synch_thread_to_empty"
       "replace-with-syncwarp" "atomic-add-to-atomic-add-block"
       "atomic-to-direct" "convert-if-else-to-if-body" "simplify-if-statements")

# Function to generate the combination of flags
generate_flag_combinations() {
    local prefix=$1
    local flags=("${!2}")
    local index=$3

    if [ $index -eq ${#flags[@]} ]; then
        echo $prefix
        return
    fi

    generate_flag_combinations "$prefix --${flags[$index]}=true" flags[@] $((index + 1))
    generate_flag_combinations "$prefix --${flags[$index]}=false" flags[@] $((index + 1))
}

# Count the number of combinations
count=0
for combination in $(generate_flag_combinations "" FLAGS[@] 0); do
    count=$((count + 1))
done

echo "Total number of combinations: $count"

# Process each .cu file in the input directory
file_count=1
for SOURCE_FILE in "$INPUT_DIR"/*.cu; do
    base_filename=$(basename -- "$SOURCE_FILE" .cu)
    
    echo "Processing file: $SOURCE_FILE"

    # Process the source file with each combination of flags
    combination_count=1
    for combination in $(generate_flag_combinations "" FLAGS[@] 0); do
        OUTPUT_FILE="$OUTPUT_DIR/${base_filename}_${combination_count}.cu"

        echo "Processing $SOURCE_FILE with flags: $combination"

        # Run the tool and save the output, with error handling
        if $TOOL_COMMAND $combination "$SOURCE_FILE" -- $TOOL_ARGS > "$OUTPUT_FILE"; then
            echo "Successfully processed $SOURCE_FILE, output saved to $OUTPUT_FILE"
        else
            echo "Error occurred while processing $SOURCE_FILE with flags: $combination"
        fi

        combination_count=$((combination_count + 1))
    done

    file_count=$((file_count + 1))
done

echo "Processing complete. Output files are saved in $OUTPUT_DIR"

