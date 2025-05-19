PYTHON_VERSION       := 3.12
PYTHON               := uv run python

MISTRAL_SCRIPT       := src/data/mistral.py
OPENAI_SCRIPT        := src/data/openai.py
TRANSLATE_SCRIPT     := src/data/translate.py
PUSH_TO_HF_SCRIPT    := src/data/push_to_hf.py
TRAIN_SCRIPT         := src/train/train.py

MISTRAL_MODEL        ?= mistral-small-2501
MISTRAL_REQUESTS     ?= 100000
OPENAI_BASE_URL      ?= https://api.openai.com/v1/chat/completions
OPENAI_MODEL         ?= gpt-3.5-turbo
OPENAI_REQUESTS      ?= 512
SPLIT                ?= train
NLLB_MODEL           ?= facebook/nllb-200-distilled-600M
BATCH_SIZE           ?= 32
FOLDER_PATH          ?= ./backup/
REPO_NAME            ?= MaxLSB/LeCarnet

.PHONY: env generate-mistral generate-openai translate push-dataset train

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
		--base_url $(OPENAI_BASE_URL) \
		--model_name $(OPENAI_MODEL) \
		--total_requests $(OPENAI_REQUESTS)

translate:
	$(PYTHON) $(TRANSLATE_SCRIPT) \
		--split $(SPLIT) \
		--model_name $(NLLB_MODEL) \
		--batch_size $(BATCH_SIZE)

push-dataset:
	$(PYTHON) $(PUSH_TO_HF_SCRIPT) \
		--folder_path $(FOLDER_PATH) \
		--repo_name $(REPO_NAME)

train:
	$(PYTHON) $(TRAIN_SCRIPT)
