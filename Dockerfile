FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="David Morrison"

# update the base image and install gosu
RUN apt-get update -y \
    && apt-get install -y build-essential \
    && apt-get -y install sudo gosu

# setup the ubuntu user
RUN groupadd ubuntu \
    && useradd -ms /bin/bash -g ubuntu ubuntu \
    && usermod -a -G sudo ubuntu

USER root
ENTRYPOINT [ "/bin/bash", "./scripts/entrypoint.sh" ]