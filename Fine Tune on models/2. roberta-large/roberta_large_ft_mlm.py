# -*- coding: utf-8 -*-

import datasets
import transformers
import pandas as pd
from datasets import Dataset

#Tokenizer
from transformers import RobertaTokenizerFast

#Encoder-Decoder Model
from transformers import EncoderDecoderModel

#Training
# When using previous version of the library you need the following two lines
#from seq2seq_trainer import Seq2SeqTrainer
#from transformers import TrainingArguments

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional

import os

if __name__ == '__main__':

    # Set the path to the data folder, datafile and output folder and files
    root_folder = os.getcwd()
    data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
    model_folder = os.path.abspath(os.path.join(root_folder, 'Projects/Fine-Tuned-Model'))
    output_folder = os.path.abspath(os.path.join(root_folder, 'Projects/Final-model'))
    # roberta_base = os.path.abspath(os.path.join(root_folder, 'Projects/roberta-base'))
    roberta_large = os.path.abspath(os.path.join(root_folder, 'Projects/roberta-large'))


    # Datafiles names containing training and test data
    test_filename='test.csv'
    datafile= 'train.csv'
    outputfile = 'submission.csv'
    datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
    testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))
    outputfile_path = os.path.abspath(os.path.join(output_folder,outputfile))

    # Load the dataset from a CSV file
    df = pd.read_csv(datafile_path, header=0, usecols=[0,1])
    print('Num Examples: ',len(df))
    print('Null Values\n', df.isna().sum())

    df.dropna(inplace=True)
    print('Num Examples: ',len(df))

    # Splitting the data into training and validation
    # Defining the train size. So 90% of the data will be used for training and the rest will be used for validation. 
    train_size = 0.9
    # Sampling 90% fo the rows from the dataset
    train_dataset=df.sample(frac=train_size,random_state = 42)
    # Reset the indexes
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print('Length Train dataset: ', len(train_dataset))
    print('Length Val dataset: ', len(val_dataset))

    # To limit the training and validation dataset, for testing
    max_train=len(train_dataset)
    max_val= len(val_dataset) # 3155
    # Create a Dataset from a pandas dataframe for training and validation
    train_data=Dataset.from_pandas(train_dataset[:max_train])
    val_data=Dataset.from_pandas(val_dataset[:max_val])


    ## === Setting the model and training parameters === ##
    TRAIN_BATCH_SIZE = 64 # input batch size for training (default: 64)
    VALID_BATCH_SIZE = 1000 # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = 40 # number of epochs to train (default: 10)
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512           # Max length for NL annotation
    SUMMARY_LEN = 512         # Max length for API Names

    # Loading the RoBERTa Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(roberta_large,  max_len=MAX_LEN)
    # Setting the BOS and EOS token
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    ## Prepare and create the dataset ##

    batch_size=TRAIN_BATCH_SIZE  # change to 16 for full training
    encoder_max_length=MAX_LEN
    decoder_max_length=SUMMARY_LEN

    def process_data_to_model_inputs(batch):
        # Tokenize the input and target data
        inputs = tokenizer(batch["annotation"], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = tokenizer(batch["api_seq"], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    # Preprocessing the training data
    train_data = train_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=train_data.column_names
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    # Preprocessing the validation data
    val_data = val_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=val_data.column_names
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    ## Define the RoBERTa Encoder-Decoder model ##

    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(roberta_large, roberta_large, tie_encoder_decoder=True)

    # Show the vocab size to check it has been loaded
    print('Vocab Size: ', roberta_shared.config.encoder.vocab_size)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params                               
    roberta_shared.config.max_length = SUMMARY_LEN
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 1
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.repetition_penalty = 3.0
    roberta_shared.config.num_beams = 10
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    
    ## Create the Trainer ##

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_folder,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        #evaluate_during_training=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024, # 1024,  
        save_strategy = "epoch",
        # save_steps=32, # 2048, 
        warmup_steps= 1024, # 1024,  
        #max_steps=1500, # delete for full training
        num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
        # overwrite_output_dir=True,
        save_total_limit=40,
        fp16=False
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=roberta_shared,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # Fine-tune the model, training and evaluating on the train dataset
    trainer.train()

    # Save the encoder-decoder model just trained
    trainer.save_model(output_folder)