FROM python:3.10

WORKDIR /app

# Copy all of OpenAVMKit's build files into the container (excluding those in .dockerignore)
COPY . ./

# --no-cache-dir is used to avoid caching packages, shrinking the image size
RUN pip install --no-cache-dir -r requirements.txt

# Install local openavmkit package
RUN pip install .

# Seperately install jupyter (as specified on openavmkit docs)
RUN pip install jupyter

# Install and register the Python kernel for Jupyter
# This makes the kernel visible to the Jupyter server.
RUN python -m ipykernel install --user --name=python3 --display-name="Python 3 (Project)"

# Expose the notebooks file with jupyter notebook on container start
# IP 0.0.0.0 sets it to be accessed external to the container
# Allow root allows it to modify the file structure and volume
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root" ]

LABEL maintainer="Jackson Arnold <jackson.n.arnold@gmail.com>"