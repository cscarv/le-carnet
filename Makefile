# Makefile
SHELL := cmd.exe

# Variables
ifeq ($(OS),Windows_NT)
    ENV_DIR ?= .venv
    ACTIVATE := $(ENV_DIR)\Scripts\activate.bat
    PIP := $(ENV_DIR)\Scripts\pip.exe
    PYTHON := $(ENV_DIR)\Scripts\python.exe
else
    ACTIVATE := $(ENV_DIR)/bin/activate
    PIP := $(ENV_DIR)/bin/pip
    PYTHON := $(ENV_DIR)/bin/python
endif

MISTRAL := src/data/mistral.py
OPENAI := src/data/openai.py
TRANSLATE_SCRIPT := src/data/translation.py

# Defaults
MISTRAL_MODEL      ?= mistral-small-2501
MISTRAL_TOTAL_REQUESTS ?= 100000

OPENAI_BASE_URL    ?= https://api.openai.com/v1/chat/completions   
OPENAI_MODEL       ?= gpt-3.5-turbo
OPENAI_TOTAL_REQUESTS   ?= 512

SPLIT              ?= train
MODEL              ?= facebook/nllb-200-distilled-600M
BATCH_SIZE         ?= 32

ENV_DIR            ?= .venv

PYTHON_PACKAGES = \
    transformers==4.51.3 \
    datasets==3.5.0 \
    sentencepiece==0.2.0 \
    sacremoses==0.1.1 \
    accelerate==1.5.1 \
    bitsandbytes==0.45.4 \
	jsonlines==4.0.0 \
	mistralai==1.7.0 \

.PHONY: generate-mistral generate-openai translate env

# Create venv only if not already present
env: $(ACTIVATE)

$(ACTIVATE):
	@echo Creating virtual environment...
	python -m venv $(ENV_DIR)
	$(PYTHON) -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124  \
        torch==2.6.0+cu124
	$(PYTHON) -m pip install $(PYTHON_PACKAGES)

# Targets
generate-mistral:
	$(PYTHON) $(MISTRAL) --model_name $(MISTRAL_MODEL) --total_requests $(MISTRAL_TOTAL_REQUESTS)

generate-openai: 
	$(PYTHON) $(OPENAI) --base_url $(OPENAI_BASE_URL) --model_name $(OPENAI_MODEL) --total_requests $(OPENAI_TOTAL_REQUESTS)

translate:
	$(PYTHON) $(TRANSLATE_SCRIPT) --split $(SPLIT) --model_name $(MODEL) --batch_size $(BATCH_SIZE)