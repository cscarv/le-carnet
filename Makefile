# Makefile

# Variables
PYTHON := python
MISTRAL := src/data/mistral.py
OPENAI := src/data/openai.py
TRANSLATE := src/data/translation.py

# Generate Stories with Mistral
.PHONY: env generate-mistral generate-openai translate

MISTRAL_MODEL ?= mistral-small-2501
TOTAL_REQUESTS ?= 512

generate-mistral:
	$(PYTHON) $(MISTRAL) --model_name $(MISTRAL_MODEL) --total_requests $(TOTAL_REQUESTS)

BASE_URL ?= https://api.openai.com/v1/chat/completions
OPENAI_MODEL ?= gpt-3.5-turbo
TOTAL_REQUESTS ?= 512

generate-openai:
	$(PYTHON) $(OPENAI) --base_url $(BASE_URL) --model_name $(OPENAI_MODEL) --total_requests $(TOTAL_REQUESTS)

SPLIT ?= train
MODEL = facebook/nllb-200-distilled-600M
BATCH_SIZE ?= 32

translate:
	$(PYTHON) $(TRANSLATE_SCRIPT) --split $(SPLIT) --model_name $(MODEL) --batch_size $(BATCH_SIZE)


ENV_DIR ?= .venv

env:
	python3 -m venv $(ENV_DIR)
	$(ENV_DIR)/bin/pip install --upgrade pip
	$(ENV_DIR)/bin/pip install transformers datasets torch sentencepiece sacremoses accelerate bitsandbytes