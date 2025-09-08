FROM python:3.10

WORKDIR /usr/src/app

## Copy all of OpenAVMKit's build files into the container
COPY . ./

VOLUME /notebooks

RUN ls

RUN pip install -r requirements.txt

RUN pip install -e .

RUN pip install jupyter

# Expose the notebooks file with jupyter notebook on container start
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root" ]