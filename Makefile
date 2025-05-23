PYTHON_VERSION       := 3.12
PYTHON               := uv run python

MISTRAL_SCRIPT       := src/data/mistral.py
OPENAI_SCRIPT        := src/data/openai.py
TRANSLATE_SCRIPT     := src/data/translate.py
PUSH_TO_HF_SCRIPT    := src/data/push_to_hf.py
TRAIN_SCRIPT         := src/train/train.py
INFERENCE_SCRIPT     := src/inference/inference.py

# Data generation parameters
MISTRAL_MODEL        ?= mistral-small-2501
MISTRAL_REQUESTS     ?= 100000
OPENAI_MODEL         ?= gpt-3.5-turbo
OPENAI_REQUESTS      ?= 100000
SPLIT                ?= train
NLLB_MODEL           ?= facebook/nllb-200-distilled-600M
FOLDER_PATH          ?= ./backup/
REPO_NAME            ?= MaxLSB/LeCarnet

# Train parameters
MODEL_REPO_NAME ?= MaxLSB/LeCarnet-16M
MODEL_CONFIG ?= 16M

# Inference parameters
MODEL_NAME ?= MaxLSB/LeCarnet-2M
MAX_NEW_TOKENS ?= 256
PROMPT ?= "Il était une fois" 

.PHONY: env generate-mistral generate-openai translate push-dataset train inference

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up environment..."
	@uv sync --python $(PYTHON_VERSION)
	@echo "Environment ready."

env-gpu: env
	@uv sync --extra gpu

generate-mistral:
	$(PYTHON) $(MISTRAL_SCRIPT) \
		--model_name $(MISTRAL_MODEL) \
		--total_requests $(MISTRAL_REQUESTS)

generate-openai:
	$(PYTHON) $(OPENAI_SCRIPT) \
		--model_name $(OPENAI_MODEL) \
		--total_requests $(OPENAI_REQUESTS)

translate:
	$(PYTHON) $(TRANSLATE_SCRIPT) \
		--split $(SPLIT) \
		--model_name $(NLLB_MODEL)

push-dataset:
	$(PYTHON) $(PUSH_TO_HF_SCRIPT) \
		--folder_path $(FOLDER_PATH) \
		--repo_name $(REPO_NAME)

train:
	$(PYTHON) $(TRAIN_SCRIPT) \
		--repo_name $(MODEL_REPO_NAME) \
		--model_config $(MODEL_CONFIG)

inference:
	$(PYTHON) $(INFERENCE_SCRIPT) \
		--model_name $(MODEL_NAME) \
		--prompt "$(PROMPT)" \
		--max_new_tokens $(MAX_NEW_TOKENS)
