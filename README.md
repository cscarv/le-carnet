# LeCarnet

<div align="center">
  <img src="./media/lecarnet.png" width="100%" alt="LeCarnet Logo" />
</div>
<br style="line-height: 12px;" />
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/datasets/MaxLSB/LeCarnet" target="_blank" style="margin: 2px;">
    <img alt="LeCarnet Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/MaxLSB/lecarnet-683d6b6843023b2c88258594" target="_blank" style="margin: 2px;">
    <img alt="LeCarnet Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=ffffff" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## 1. Introduction

**LeCarnet** is a text dataset of **2 million** children's stories in **french** using very simple vocabulary, inspired by the English TinyStories dataset. 
The purpose of this work is to provide a reliable, high-quality resource for training and evaluating small language models (SLMs). It is aimed at educational and experimental use. This repository contains the data generation pipeline, as well as the training, evaluation, and inference code that we used.

This dataset was created by synthetically generating French short stories using [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

The dataset and models are available on Hugging Face:
- [LeCarnet Dataset](https://huggingface.co/datasets/MaxLSB/LeCarnet)
- [LeCarnet-3M](https://huggingface.co/MaxLSB/LeCarnet-3M)
- [LeCarnet-8M](https://huggingface.co/MaxLSB/LeCarnet-8M)
- [LeCarnet-21M](https://huggingface.co/MaxLSB/LeCarnet-21M)

## 2. Quick Setup

_Using [`uv`](https://github.com/astral-sh/uv) for fast and reliable dependency management._

```bash
# Basic environment setup
make env

# Basic environment setup (for GPUs)
make env-gpu
```
That's it, you can now run any command you want!

## 3. Training & Inference
The training pipeline supports Weights & Biases (WandB) for tracking training and validation losses, as well as perplexity.

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Training    | `make train`       | `python src/train/train.py --model_config MODEL_CONFIG`                                                                                 | `MODEL_CONFIG=3M`                              |
| Inference   | `make inference`   | `python src/inference/inference.py --model_name MODEL_NAME --prompt PROMPT --max_new_tokens MAX_NEW_TOKENS`                                              | `MODEL_NAME=MaxLSB/LeCarnet-3M`, `PROMPT="Il était une fois"`, `MAX_NEW_TOKENS=512` |
| Push Model to HF   | `make push-model`   | `python src/inference/push-model.py --repo_name HF_REPO --model_dir MODEL_DIR`                                              | `HF_REPO=MaxLSB/LeCarnet-3M`, `MODEL_DIR=LeCarnet-3M/model_weights/` |

_Check `src/train/configs.py` for fine-grained hyperparameter tuning. MODEL_CONFIG="custom" to use your own custom model config._

## 4. Data Generation
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

| Task                          | Make Command           | Equivalent CLI Command                                                                                                                                               | Default Values                                                                                     |
|-------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Generate with Mistral         | `make generate-mistral`| `python src/data/mistral.py --model_name MISTRAL_MODEL --total_requests MISTRAL_REQUESTS, --num_workers NUM_WORKERS`                                              | `MISTRAL_MODEL=mistral-small-2501`, `MISTRAL_REQUESTS=100000`, `NUM_WORKERS=2`                                               |
| Generate with OpenAI          | `make generate-openai` | `python src/data/openai.py --model_name OPENAI_MODEL --total_requests OPENAI_REQUESTS`                                                   | `OPENAI_MODEL=gpt-3.5-turbo`, `OPENAI_REQUESTS=100000`                                                    |
| Push Dataset to HF            | `make push-dataset`    | `python src/data/push_dataset.py --folder_path FOLDER_PATH --repo_name REPO_NAME`                                           | `FOLDER_PATH=./dataset/`, `REPO_NAME=MaxLSB/LeCarnet`                                                |

_Not all arguments are listed here._

## 4. References

- [`TinyStories: How Small Can Language Models Be and Still Speak Coherent English?`](https://arxiv.org/pdf/2305.07759)
- [`Regional Tiny Stories: Using Small Models to Compare Language Learning and Tokenizer Performance`](https://arxiv.org/pdf/2504.07989)
