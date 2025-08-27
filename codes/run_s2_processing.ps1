# run_s2_processing.ps1 (Full, Final Version)

# This script automates the process of running s2_prepare_dl_data.py on all Parquet chunks.
# It will create a separate output folder (e.g., processed_for_dl_1, processed_for_dl_2) for each chunk.

Write-Host "--- Starting batch processing of Parquet chunks ---" -ForegroundColor Green

# --- USER CONFIGURATION ---
# IMPORTANT: Update these two variables to match your dataset.
# The base name of your dataset from the s1 script output
$baseFilename = "Prepared_Dataset_1756200284" 
# The total number of chunks you created with the s1b script
$numChunks = 10 
# --- END OF CONFIGURATION ---


# Loop from 1 to the total number of chunks
1..$numChunks | ForEach-Object {
    $i = $_
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "--- Processing Chunk $i/$numChunks ---" -ForegroundColor Cyan
    Write-Host "================================================="

    # Construct the input filename and a simpler output suffix for the new folder name
    $inputFile = "${baseFilename}_part_${i}_of_${numChunks}.parquet"
    $outputSuffix = "_${i}"
    
    # Run the Python script with the correct arguments
    python codes/s2_prepare_dl_data.py --input_file $inputFile --output_suffix $outputSuffix
    
    # Check if the last command failed
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
        Write-Host "!!! ERROR: An error occurred while processing chunk $i." -ForegroundColor Red
        Write-Host "!!! Aborting script." -ForegroundColor Red
        Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
        exit 1 # Exit the script immediately on failure
    }
}

Write-Host ""
Write-Host "--- All chunks processed successfully ---" -ForegroundColor Green
Write-Host "You are now ready to run the merge script (s2b_merge_chunks.py)." -ForegroundColor Green