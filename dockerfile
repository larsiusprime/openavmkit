# Stage 1: Build the OpenAVMKit package and install dependencies
FROM python:3.10 as builder

WORKDIR /app

# Copy all of OpenAVMKit's build files into the container (excluding those in .dockerignore)
COPY . ./

# --no-cache-dir is used to avoid caching packages, shrinking the image size
RUN pip install --no-cache-dir -r requirements.txt

# Install local openavmkit package
RUN pip install .

# Seperately install jupyter (as specified on openavmkit docs)
RUN pip install jupyter



# Stage 2: Muve the built/installed packages into a distroless environment
# Install and register the Python kernel for Jupyter
# This makes the kernel visible to the Jupyter server.
RUN python -m ipykernel install --user --name=python3 --display-name="Python 3 (Project)"

# Expose the notebooks file with jupyter notebook on container start
# IP 0.0.0.0 sets it to be accessed external to the container
# Allow root allows it to modify the file structure and volume
# --no-browser avoids opening the browser automatically, as there is not one in the container
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root", "--no-browser" ]

LABEL maintainer="Jackson Arnold <jackson.n.arnold@gmail.com>"

# Future updates:
# - Create all the dependencies in a distro environment, then move it to a distroless with the root file being /notebooks/ (no need for anything outside of that)