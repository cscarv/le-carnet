# LeCarnet: the french TinyStories

_(Work in progress...)_

LeCarnet is a text dataset of 1 million children's stories in french using very simple vocabulary, inspired by the English TinyStories dataset. 
The purpose is to provide a reliable, high-quality resource for training and evaluating small language models (SLMs). It is aimed at educational and experimental use.

This dataset was created by synthetically generating French short stories using [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

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

| Task                          | Make Command           | Equivalent CLI Command                                                                                                                                               |
|-------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Training | `make train` | `python src/train/train.py --dataset_name MaxLSB/LeCarnet --train_batch_size 16 --gradient_accumulation_steps 4 --learning_rate 5e-4 --max_train_steps 10000`                                                |
| Inference   | `make inference` | `python src/inference/inference.py --model_name MaxLSB/LeCarnet-3M --prompt "Il était une fois" --max_new_tokens 256`                                                |

_Not all arguments are listed here._

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

| Task                          | Make Command           | Equivalent CLI Command                                                                                                                                               |
|-------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Generate with Mistral    | `make generate-mistral` | `python src/data/mistral.py --model_name mistral-small-2501 --total_requests 512`                                                |
| Generate with OpenAI       | `make generate-openai`  | `python src/data/openai.py --base_url https://api.openai.com/v1/chat/completions --model_name gpt-3.5-turbo --total_requests 512` |
| Translate TinyStories Dataset | `make translate`        | `python src/data/translation.py --split train --model_name facebook/nllb-200-distilled-600M --batch_size 32 `                              |
| Push Dataset to HF | `make push-dataset`        | `python src/data/push_to_hf.py --folder_path ./backup/ --repo_name MaxLSB/LeCarnet`                   |

_Not all arguments are listed here._

# References

- [`TinyStories: How Small Can Language Models Be and Still Speak Coherent English?`](https://arxiv.org/pdf/2305.07759)
- [`TinyStories Dataset`](https://huggingface.co/datasets/roneneldan/TinyStories)
