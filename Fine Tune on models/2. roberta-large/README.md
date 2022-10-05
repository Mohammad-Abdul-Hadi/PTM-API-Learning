# RoBERTa base model

Pretrained model on English language using a masked language modeling (MLM) objective. This model is case-sensitive: it
makes a difference between english and English.

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. 

### How to use

You can use this model directly on the training dataset for finetuning:

1. get the dataset from the '../dataset' folder; Put them in 'dataset/' folder

2. Run the following commands from the current directory:

```
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install torch torchvision torchaudio
python3 roberta_large_ft_mlm.py
```

3. After Fine-tuning, the models will be saved in the 'Projects/' folder.