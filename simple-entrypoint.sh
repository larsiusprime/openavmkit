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

    echo "--- Starting Notebook Test Run ---"
    
    echo "Running: 01-assemble.ipynb"
    jupyter execute notebooks/pipeline/01-assemble.ipynb --kernel python3
    
    echo "Running: 02-clean.ipynb"
    jupyter execute notebooks/pipeline/02-clean.ipynb --kernel python3
    
    echo "Running: 03-model.ipynb"
    jupyter execute notebooks/pipeline/03-model.ipynb --kernel python3

    # Notebooks 04 and 05 to be added when they are complete

    echo "Running: assessment-quality.ipynb"
    jupyter execute notebooks/pipeline/assessment-quality.ipynb --kernel python3
    
    echo "--- All notebooks ran successfully ---"
else
    # Expose the notebooks file with jupyter notebook on container start
    # IP 0.0.0.0 sets it to be accessed external to the container
    # Allow root allows it to modify the file structure and volume
    # --no-browser avoids opening the browser automatically, as there is not one in the container
    exec jupyter notebook --ip 0.0.0.0 --allow-root --no-browser  
fi