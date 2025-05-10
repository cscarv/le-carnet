# French TinyStories

French TinyStories (name to be changed) is a text dataset of 1 million children's stories in french using very simple vocabulary, based on the English TinyStories dataset. 
The purpose is to provide a reliable, high-quality resource for pretraining small language models from scratch, aimed at educational and experimental use.

"This dataset was created by synthetically generating French short stories using Mistral Small 3.1.

_Validation is done (12 000 samples)! Currently working on the training set._

## Dataset

- Model used for generation: 
    - [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
    - [`mistralai/Mistral-Nemo-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- Model used for translation: [`facebook/nllb-200-1.3B`](https://huggingface.co/facebook/nllb-200-1.3B)

## Objectives

- Build a TinyStories dataset in French (1M samples for training, 12k samples for validation)
- Train models from scratch (1M / 3M / 8M / 28M parameters) on the dataset
- Train multilingual models (French + English) for TinyStories generation

## Building the Dataset
_Commands must be executed from the project root directory._

Generate batch of stories using an API:
```bash
echo "your_api_key_here" > api_key.txt
```
```bash
python generate.py --base_url "your_base_url" --model_name "your_model_name" --total_requests 2048 --batch_size 32 --concurrency 2
```

Generate stories with Mistral (Free Tier):
```bash
echo "your_api_key_here" > mistral_api_key.txt
```
```bash
python mistral.py --api_key_file "mistral_api_key.txt" --model_name "mistral-small-2501" --total_requests 2048
```

Translate stories from TinyStories:
```bash
python translate.py --split train --batch_size 64 --num_workers 4
```

Push the dataset to HF in parquet format:
```bash
python hugging_face.py --file_path "data/my_custom_file.jsonl" --splits "validation" --repo_name "username/my-custom-repo"
```

# References

- [`TinyStories: How Small Can Language Models Be and Still Speak Coherent English?`](https://arxiv.org/pdf/2305.07759)
- [`TinyStories Dataset`](https://huggingface.co/datasets/roneneldan/TinyStories)
