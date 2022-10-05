import datasets
import torch
import transformers
from transformers import RobertaTokenizerFast, EncoderDecoderModel
import pandas as pd
from datasets import Dataset
import sentencepiece
import configparser
import sys, os, json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Encoder-Decoder Model

# Set the path to the data folder, datafile and output folder and files
root_folder = os.getcwd()
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
model_folder = os.path.abspath(os.path.join(root_folder, 'model/final-checkpoint'))
output_folder = os.path.abspath(os.path.join(root_folder, 'output'))

# Datafiles names containing training and test data
test_filename='test.csv'
datafile = 'evaluation.csv'
outputfile = 'submission.json'
datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))
outputfile_path = os.path.abspath(os.path.join(output_folder,outputfile))

"""# TEST ON THE TEST DATA"""

config = configparser.ConfigParser()
config.read('config.ini')
TRAIN_BATCH_SIZE = int(config['Model_Configuration']['TRAIN_BATCH_SIZE'])
MAX_LEN = int(config['Model_Configuration']['MAX_LEN'])
batch_size = TRAIN_BATCH_SIZE
l = int(config['Model_Configuration']['load'])
ds_size = int(config['Model_Configuration']['dataset_size'])
weights = list(map(float, config['Model_Configuration']['weights'].split(", ")))
num_seq = int(config['Model_Configuration']['num_of_ret_seq'])

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizerFast.from_pretrained(model_folder)
model = EncoderDecoderModel.from_pretrained(model_folder)

# Generate a text using beams search
def generate_summary_beam_search(batch):

    inputs = tokenizer(batch["annotation"], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = inputs.input_ids.to(torch_device)
    attention_mask = inputs.attention_mask.to(torch_device)

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                num_beams=15,
                                repetition_penalty=3.0, 
                                length_penalty=2.0, 
                                num_return_sequences = num_seq
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

# Generate a text using beams search
def generate_summary_topk(batch):
    inputs = tokenizer(batch["annotation"], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = inputs.input_ids.to(torch_device)
    attention_mask = inputs.attention_mask.to(torch_device)

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                repetition_penalty=3.0, 
                                length_penalty=2.0, 
                                num_return_sequences = num_seq,
                                do_sample=True,
                                top_k=50, 
                                top_p=0.95,

    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

def test():
    # Load the dataset:  
    df = pd.read_csv(testfile_path, header=0)

    test_data = Dataset.from_pandas(df) #.head(ds_size))
    
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    model.to(torch_device)  
    print("===== Prediction Started =====")
    # Generate predictions using top-k sampling
    results = test_data.map(generate_summary_topk, batched=True, batch_size=batch_size, remove_columns=["annotation"])
    
    sys.stdout = old_stdout # reset old stdout

    print("===== Prediction Saved =====")
    pred_str_topk = results["pred"]

    json_string = json.dumps(pred_str_topk)

    with open(outputfile_path, "w") as outfile:
        outfile.write(json_string)

    print("===== Evaluation Started =====")
    df = pd.read_csv(datafile_path, header=0)
    cc = SmoothingFunction()

    bleu_scores = 0
    for i in range(len(pred_str_topk[:ds_size])):
        reference = pred_str_topk[i].split()
        candidate = df['api_seq'][i // num_seq].split()
        bleu_scores += sentence_bleu(reference, candidate, smoothing_function=cc.method7, weights = weights)*l

    bleu_score = (bleu_scores/len(pred_str_topk[:ds_size]))
    print("Bleu Score:", bleu_score)

if __name__ == "__main__":
    test()