#################################################################################
# CONSTANTS                                                                     #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = pathgen
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.6

UID = $(shell id -u)
GID = $(shell id -g)

# network
JUPYTER_PORT := 8800

#################################################################################
# PYTHON ENVIRONMENT COMMANDS                                                   #
#################################################################################
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	echo "source activate $(PROJECT_NAME)" > ~/.bashrc
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"

requirements:
	conda env update --file environment.yml
	pip install -r requirements.txt

install_openslide:
	sudo apt-get update
	sudo apt install -y build-essential
	sudo apt-get -y install openslide-tools
	pip install Pillow
	pip install openslide-python

install_opencv:
	sudo apt-get update && apt-get install -y python3-opencv

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build --build-arg UID=$(UID) --build-arg  GID=$(GID) -t $(PROJECT_NAME) .

docker_run:
	docker run --shm-size=16G \
				--gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/ubuntu/$(PROJECT_NAME) \
				-v /raid/datasets:/home/ubuntu/$(PROJECT_NAME)/data \
				-v /raid/experiments/$(PROJECT_NAME):/home/ubuntu/$(PROJECT_NAME)/experiments \
				-it $(PROJECT_NAME):latest

#################################################################################
# JUPYTER COMMANDS                                                              #
#################################################################################
setup_jupyter:
	pip install --user ipykernel
	python -m ipykernel install --user --name=pathgen

run_notebook:
	jupyter notebook --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root
