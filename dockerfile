# Stage 1: Build the OpenAVMKit package and install dependencies
FROM python:3.10 AS builder

WORKDIR /app

# Copy all of OpenAVMKit's build files into the container (excluding those in .dockerignore)
COPY . ./

# Ensure that the entrypoint script is executable
RUN chmod +x ./simple-entrypoint.sh

# --no-cache-dir is used to avoid caching packages, shrinking the image size
RUN pip install --no-cache-dir -r requirements.txt

# Install local openavmkit package
RUN pip install .

# Seperately install jupyter (as specified on the openavmkit docs)
RUN pip install jupyter

# Stage 2: Muve the built/installed packages into a distroless environment

# Install and register a standard Python kernel for Jupyter
RUN python -m ipykernel install --user --name=python3 --display-name="Python 3 (Project)"

ENTRYPOINT ["./simple-entrypoint.sh"]

LABEL maintainer="Jackson Arnold <jackson.n.arnold@gmail.com>"

# Future updates:
# - Create all the dependencies in a distro environment, then move it to a distroless with the root file being /notebooks/ (no need for anything outside of that)