#!/bin/bash

# If any of the jupyter execute commands fail, exit the script
set -e

if [ "$RUN" = "test" ]; then
    # Create /app/notebooks/pipeline/data/us-nc-guilford/ directory
    mkdir -p /app/notebooks/pipeline/data/us-nc-guilford/

    # Create the cloud.json file in the /app/notebooks/pipeline/data/us-nc-guilford/ directory
    echo '{
        "type": "azure",
        "azure_storage_container_url": "https://landeconomics.blob.core.windows.net/localities-public"
    }' > /app/notebooks/pipeline/data/us-nc-guilford/cloud.json

    # Run all finished notebooks to test them on (currently) guilford
    # nbconvert is being used instead of execute because only it has
    # the flag to specify a kernel
    # Flags ensure that the notebook runs on an existing kernel, nbconvert
    # executes the notebook before conversion and
    # that it doesn't produce output that clutters the source directory
    echo "--- Starting Notebook Test Run on $LOCALITY ---"
    
    echo "Running: 01-assemble.ipynb"
    jupyter nbconvert \
        --to notebook \
        --inplace \
        --execute \
        --ExecutePreprocessor.kernel_name=python3 \
        notebooks/pipeline/01-assemble.ipynb
    
    echo "Running: 02-clean.ipynb"
    jupyter nbconvert \
        --to notebook \
        --inplace \
        --execute \
        --ExecutePreprocessor.kernel_name=python3 \
        notebooks/pipeline/02-clean.ipynb
    
    echo "Running: 03-model.ipynb"
    jupyter nbconvert \
        --to notebook \
        --inplace \
        --execute \
        --ExecutePreprocessor.kernel_name=python3 \
        notebooks/pipeline/03-model.ipynb

    # Notebooks 04 and 05 to be added when they are complete

    echo "Running: assessment-quality.ipynb"
    jupyter nbconvert \
        --to notebook \
        --inplace \
        --execute \
        --ExecutePreprocessor.kernel_name=python3 \
        notebooks/pipeline/assessment-quality.ipynb
    
    echo "--- All notebooks ran successfully ---"
else
    # Expose the notebooks file with jupyter notebook on container start
    # IP 0.0.0.0 sets it to be accessed external to the container
    # Allow root allows it to modify the file structure and volume
    # --no-browser avoids opening the browser automatically, as there is not one in the container
    exec jupyter notebook --ip 0.0.0.0 --allow-root --no-browser  
fi