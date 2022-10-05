# Deep API Learning

It is a PyTorch implementation of deepAPI. See [Deep API Learning](https://arxiv.org/abs/1605.08535) for more details. 

## Prerequisites
 - PyTorch 0.4
 - Python 3.6
 - Numpy
 

## Usage

### Dataset
download data from [Google Driver](https://drive.google.com/drive/folders/1jBKMWZr5ZEyLaLgH34M7AjJ2v52Cq5vv?usp=sharing) and save them to the `./data` folder

### Train
   `$ python train.py`
will run default training and save model to ./output

### Test

Then you can run the model by:

    python sample.py
    
The outputs will be printed to stdout and generated responses will be saved at results.txt in the `./output/` path.