import torch
from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":
    model = AutoModel.from_pretrained() # Specify Updated Model Path 
    tokenizer = AutoTokenizer.from_pretrained() # Specify Updated Model Tokenizer Path

    new_words = [] # list of API_Names that needs to be incorporated in the PTM vocabulary 
    counter = len(new_words)# num_of_added_tokens

    with torch.no_grad():
        for i in range(1,  counter+1):
            tokenized_text = tokenizer.tokenize(new_words[(-1)*i])
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # print(indexed_tokens)
            t= []
            for it in indexed_tokens:
                t.append(model.embeddings.word_embeddings.weight[it, :].data)
    
            t = torch.stack(t)
            avg_tensor = torch.mean(t, axis = 0)
    
            model.embeddings.word_embeddings.weight[(-1)*i, :] = avg_tensor