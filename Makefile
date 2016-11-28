.PHONY: all install test notebooks

# Support for Conda Environments

CONDA_HOME = $(HOME)/.anaconda3
CONDA_BIN_DIR = $(CONDA_HOME)/bin
CONDA = $(CONDA_BIN_DIR)/conda

ENV_NAME = "varcompfa"
ENV_DIR = $(CONDA_HOME)/envs/$(ENV_NAME)
ENV_BIN_DIR = $(ENV_DIR)/bin
ENV_LIB_DIR = $(ENV_DIR)/lib
ENV_PYTHON = $(ENV_BIN_DIR)/python

all:
	@echo "Add things to this Makefile"

install:
	pip install -r requirements.txt

test:
	py.test
	
# Convert Jupyter Notebooks to HTML
notebooks:
	export PATH='$(CONDA_BIN_DIR):$(CONDA_HOME):$$PATH'
	export CONDA_PREFIX='$(ENV_DIR)'
	@echo $$PATH
	source activate $(ENV_NAME)
	python --version
	
	$(ENV_BIN_DIR)/jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=python --to=html --FilesWriter.build_directory=./notebooks ./notebooks/*.ipynb
