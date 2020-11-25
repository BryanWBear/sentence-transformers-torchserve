FROM pytorch/torchserve:0.1-cpu

USER root

RUN pip install sentence-transformers
RUN pip install torchserve
RUN pip install torch-model-archiver -q

RUN apt-get update
RUN apt install zip unzip

USER model-server

COPY ./src .

RUN python download.py

RUN ./create-mar.sh

# ENTRYPOINT [ "torchserve","--start", "--model-store model-store", "--models sentence_xformer.mar"]