import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as Func

if __name__ == "__main__":
    model = AutoModel.from_pretrained() # Specify Updated Model Path 
    tokenizer = AutoTokenizer.from_pretrained() # Specify Updated Model Tokenizer Path

    new_words = [] # list of API_Names that needs to be incorporated in the PTM vocabulary 
    counter = len(new_words)# num_of_added_tokens

    x = torch.tensor(list(range(0, counter, 1)))
    t = Func.one_hot(x, num_classes=model.config.hidden_size)

    with torch.no_grad():
        for i in range(1,  counter+1):
            model.embeddings.word_embeddings.weight[(-1)*i, :] = t[(-1)*i]