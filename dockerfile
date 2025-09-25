FROM python:3.10

# Default PUID and PGID values (can be overridden at runtime). Use these to
# ensure the files on the volume have the permissions you need.
ENV PUID=1000
ENV PGID=1000

WORKDIR /app

## Copy all of OpenAVMKit's build files into the container
COPY . ./

RUN pip install -r requirements.txt

RUN pip install -e .

RUN pip install jupyter

# Expose the notebooks file with jupyter notebook on container start
# IP 0.0.0.0 sets it to be accessed external to the container
# Allow root allows it to modify the file structure and volume
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root" ]

LABEL maintainer="Jackson Arnold <jackson.n.arnold@gmail.com>"