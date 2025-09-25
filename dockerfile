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
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root" ]

LABEL maintainer="Jackson Arnold <jackson.n.arnold@gmail.com>"