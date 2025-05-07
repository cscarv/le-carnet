# French TinyStories

French TinyStories is a text dataset of 2 million children's stories in french using very simple vocabulary, based on the English TinyStories dataset.  

The purpose is to provide a reliable, high-quality resource for pretraining small language models from scratch, aimed at educational and experimental use.  
This dataset is built by translating the English TinyStories dataset using Facebook’s NLLB model.

_Validation set has already been translated! Currently working on the training set._

## Dataset

- Translation model used: [`facebook/nllb-200-1.3B`](https://huggingface.co/facebook/nllb-200-1.3B)

## Objectives

- Build a TinyStories dataset in French (2M samples for training, 22k samples for validation)
- Train models from scratch (1M / 3M / 8M / 28M parameters) on the dataset
- Train multilingual models (French + English) for TinyStories generation

## Building the Dataset

To translate:
```bash
python main.py --splits train --batch_size 64 --num_workers 4
```

To backup:
```bash
python save_backup.py
```

# References

- Source: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
