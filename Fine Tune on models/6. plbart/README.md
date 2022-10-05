# PLBART (base-sized model) 


## Intended uses & limitations

This repository contains the pre-trained model only, so you can use this model for (among other tasks) masked span prediction, as shown in the code example below. However, the main use of this model is to fine-tune it for a downstream task of interest, such as:
* code summarization
* code generation
* code translation
* code refinement
* code defect detection
* code clone detection. 

## Fine-tuning

1. get the dataset from the '../dataset' folder; Put them in 'dataset/' folder

2. Run the following commands from the current directory:

```
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install torch torchvision torchaudio
python3 plbart_base_ft_mlm.py
```

3. After Fine-tuning, the models will be saved in the 'Projects/' folder.