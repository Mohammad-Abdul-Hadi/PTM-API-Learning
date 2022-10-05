# CodeT5 (base-sized model) 

Pre-trained CodeT5 model. It was introduced in the paper [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models
for Code Understanding and Generation](https://arxiv.org/abs/2109.00859) by Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi and first released in [this repository](https://github.com/salesforce/CodeT5). 


## Intended uses & limitations

This repository contains the pre-trained model only, so you can use this model for (among other tasks) masked span prediction, as shown in the code example below. However, the main use of this model is to fine-tune it for a downstream task of interest, such as:
* code summarization
* code generation
* code translation
* code refinement
* code defect detection
* code clone detection. 


### How to use

Here is how to use this model:

```python
from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=8)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "{user.username}"
```

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
python3 codet5_base_ft_mlm.py
```

3. After Fine-tuning, the models will be saved in the 'Projects/' folder.