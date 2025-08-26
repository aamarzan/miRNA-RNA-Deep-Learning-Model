#!/bin/bash

# This script automates the process of running s2_prepare_dl_data.py on all Parquet chunks.

echo "--- Starting batch processing of Parquet chunks ---"

# The base name of your dataset from the s1 script output
BASE_FILENAME="Prepared_Dataset_1756200284"

# Loop from 1 to 10
for i in {1..10}
do
    echo ""
    echo "================================================="
    echo "--- Processing Chunk ${i}/10 ---"
    echo "================================================="

    # Construct the input filename and output suffix for the current chunk
    INPUT_FILE="${BASE_FILENAME}_part_${i}_of_10.parquet"
    OUTPUT_SUFFIX="_chunk${i}"

    # Run the Python script with the correct arguments
    python codes/s2_prepare_dl_data.py --input_file "$INPUT_FILE" --output_suffix "$OUTPUT_SUFFIX"

    # Check if the last command failed
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!! ERROR: An error occurred while processing chunk ${i}."
        echo "!!! Aborting script."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
done

echo ""
echo "--- All chunks processed successfully ---"
echo "You are now ready to run the merge script."