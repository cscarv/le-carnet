# French TinyStories

_(Work in progress...)_

French TinyStories (name to be changed) is a text dataset of 1 million children's stories in french using very simple vocabulary, based on the English TinyStories dataset. 
The purpose is to provide a reliable, high-quality resource for pretraining small language models from scratch, aimed at educational and experimental use.

"This dataset was created by synthetically generating French short stories using Mistral Small 3.1.



## Setup

```bash
make env
```
```bash
source .venv/bin/activate
```


## Data Generation Commands
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
| Mistral Story Generation      | `make generate-mistral` | `python src/data/mistral.py --model_name mistral-small-2501 --total_requests 512`                                                |
| OpenAI Story Generation       | `make generate-openai`  | `python src/data/openai.py --base_url https://api.openai.com/v1/chat/completions --model_name gpt-3.5-turbo --total_requests 512` |
| Translate TinyStories Dataset | `make translate`        | `python src/data/translation.py --split train --model_name facebook/nllb-200-distilled-600M --batch_size 32 `                              |


## Objectives

- Build a TinyStories dataset in French (1M samples for training, 12k samples for validation)
- Train models from scratch (1M / 3M / 8M / 28M parameters) on the dataset
- Train multilingual models (French + English) for TinyStories generation

# References

- [`TinyStories: How Small Can Language Models Be and Still Speak Coherent English?`](https://arxiv.org/pdf/2305.07759)
- [`TinyStories Dataset`](https://huggingface.co/datasets/roneneldan/TinyStories)
