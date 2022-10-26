from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    model = AutoModel.from_pretrained() # Specify Model Path 
    tokenizer = AutoTokenizer.from_pretrained() # Specify Model Tokenizer Path

    # get the current vocabulary
    vocabulary = tokenizer.get_vocab().keys()

    new_words = [] # list of API_Names that needs to be incorporated in the PTM vocabulary 
    for word in new_words:
        tokenizer.add_tokens(word)

    model.resize_token_embeddings(len(tokenizer))