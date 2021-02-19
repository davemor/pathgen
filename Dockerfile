# FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvcr.io/nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

# pass these into build using
# --build-arg UID=$(id -u) --build-arg GID=$(id -g)
# using 1337 as a default so that it can be seen easily if used
# note - these are reset to the current user by the entry point
# script.
ARG UID=1337
ARG GID=1337

LABEL maintainer="David Morrison"

# scripts updates to base image
# install sudo and gosu tools
RUN \
    apt-get update -y \
    && apt-get install -y \
    && apt-get install -y build-essential \
    && apt-get -y install sudo gosu

# Add user ubuntu with no password, add to sudo group
RUN groupadd -g $GID -o ubuntu
RUN adduser --uid $UID --gid $GID --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
RUN chmod a+rwx /home/ubuntu/

# Install Anaconda
WORKDIR "/tmp"
RUN \
    sudo apt-get update \
    && sudo apt-get install -y curl \
    && curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
    && bash Anaconda3-2020.02-Linux-x86_64.sh -b \
    && rm Anaconda3-2020.02-Linux-x86_64.sh
ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN \
    conda update conda \
    && conda update anaconda \
    && conda update --all

# mount cwd to the project dir in the container
# allow the ubuntu user to own and thus write to it
ADD --chown=ubuntu . /home/ubuntu/pathgen
WORKDIR "/home/ubuntu/pathgen"

# set up the pathgen conda environment
SHELL ["/bin/bash", "-c"]
RUN make create_environment
RUN conda init bash
RUN echo "source activate pathgen" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# install the projects dependencies
RUN make requirements
RUN make install_asap
RUN make install_openslide

# we are going to log in as root and then run the setup script
USER root
ENTRYPOINT ["/bin/bash", "./scripts/entrypoint.sh"]