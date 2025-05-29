# LeCarnet: the french TinyStories

![LeCarnet](./media/lecarnet.png)

**LeCarnet** is a text dataset of **1.4 million** children's stories in **french** using very simple vocabulary, inspired by the English TinyStories dataset. 
The purpose of this work is to provide a reliable, high-quality resource for training and evaluating small language models (SLMs). It is aimed at educational and experimental use. This repository contains the data generation pipeline, as well as the training, evaluation, and inference code that we used.

This dataset was created by synthetically generating French short stories using [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

The dataset and models are available on Hugging Face:
- [LeCarnet Dataset](https://huggingface.co/datasets/MaxLSB/LeCarnet)
- [LeCarnet-3M](https://huggingface.co/MaxLSB/LeCarnet-3M)
- [LeCarnet-8M](https://huggingface.co/MaxLSB/LeCarnet-8M)
- [LeCarnet-21M](https://huggingface.co/MaxLSB/LeCarnet-21M)

## Quick Setup

_Using [`uv`](https://github.com/astral-sh/uv) for fast and reliable dependency management._

```bash
# Basic environment setup
make env

# Basic environment setup (for GPUs)
make env-gpu
```
That's it, you can now run any command you want!

## Training & Inference
The training pipeline supports Weights & Biases (WandB) for tracking training and validation losses, as well as perplexity.

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Training    | `make train`       | `python src/train/train.py --model_config MODEL_CONFIG`                                                                                 | `MODEL_CONFIG=3M`                              |
| Inference   | `make inference`   | `python src/inference/inference.py --model_name MODEL_NAME --prompt PROMPT --max_new_tokens MAX_NEW_TOKENS`                                              | `MODEL_NAME=MaxLSB/LeCarnet-3M`, `PROMPT="Il était une fois"`, `MAX_NEW_TOKENS=512` |
| Push Model to HF   | `make push-model`   | `python src/inference/push-model.py --repo_name HF_REPO --model_dir MODEL_DIR`                                              | `HF_REPO=MaxLSB/LeCarnet-3M`, `MODEL_DIR=checkpoints/3M` |

_Check `src/train/train_config.py` for fine-grained hyperparameter tuning._

## Data Generation
For Generation tasks set your API key (for translation the model runs locally):

**Linux/MacOS:**
```bash
export MISTRAL_API_KEY=your_api_key
```
```bash
export OPENAI_API_KEY=your_api_key
```
**Windows:**
```bash
$env:MISTRAL_API_KEY="your_api_key"
```
```bash
$env:OPENAI_API_KEY="your_api_key"
```

A translation pipeline is available as we experimented with it at first, but chose to focus on the generation from scratch using Mistral.

| Task                          | Make Command           | Equivalent CLI Command                                                                                                                                               | Default Values                                                                                     |
|-------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Generate with Mistral         | `make generate-mistral`| `python src/data/mistral.py --model_name MISTRAL_MODEL --total_requests MISTRAL_REQUESTS, --num_workers NUM_WORKERS`                                              | `MISTRAL_MODEL=mistral-small-2501`, `MISTRAL_REQUESTS=100000`, `NUM_WORKERS=1`                                               |
| Generate with OpenAI          | `make generate-openai` | `python src/data/openai.py --model_name OPENAI_MODEL --total_requests OPENAI_REQUESTS`                                                   | `OPENAI_MODEL=gpt-3.5-turbo`, `OPENAI_REQUESTS=100000`                                                    |
| Translate TinyStories Dataset | `make translate`       | `python src/data/translation.py --split SPLIT --model_name NLLB_MODEL`                  | `SPLIT=train`, `NLLB_MODEL=facebook/nllb-200-distilled-600M`                      |
| Push Dataset to HF            | `make push-dataset`    | `python src/data/push_dataset.py --folder_path FOLDER_PATH --repo_name REPO_NAME`                                           | `FOLDER_PATH=./backup/`, `REPO_NAME=MaxLSB/LeCarnet`                                                |

_Not all arguments are listed here._

# References

- [`TinyStories: How Small Can Language Models Be and Still Speak Coherent English?`](https://arxiv.org/pdf/2305.07759)
