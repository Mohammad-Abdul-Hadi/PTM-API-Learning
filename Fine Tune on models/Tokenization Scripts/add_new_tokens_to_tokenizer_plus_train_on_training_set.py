import pandas as pd 
from transformers import AutoTokenizer

def get_training_corpus():
    dataset = [] # list of API_Names in the training corpus that needs to be incorporated in the PTM vocabulary 
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples

if __name__ == "__main__":
    old_tokenizer = AutoTokenizer.from_pretrained("") # Specify Model Tokenizer Path
    training_corpus = get_training_corpus()
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    tokenizer.save_pretrained() # Specify Model Tokenizer Path