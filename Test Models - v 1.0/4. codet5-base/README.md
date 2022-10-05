### How to use

You can test this model directly on the testing dataset (the test dataset is available at "datasets/test.csv"):

1. Run the following command from the current directory for initializaing a virtual environment. This will create a folder, *api-env* in the current directory:

```
python -m venv api-env
source api-env/bin/activate (for linux)
api-env\Scripts\activate (for windows)
python -m pip install --upgrade pip
```
2. Run the following command from the current directory for installing the requirements within the virtual environment.
```
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio
```
3. Finally, evaluate the model on the test data by running the following command:
```
python evaluate.py
decativate
```