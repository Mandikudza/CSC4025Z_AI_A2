Music Genre Classification — How to reproduce experiments


This submission contains the code and preprocessed files needed to reproduce the experiments (note: the full/raw Kaggle dataset is not included). 
The specific contents of this submission include:

- Cleaned/preprocessed CSVs under `dataset/cleaned_data/` (e.g.,  `X_test_clean.csv`, `y_test_clean.csv`).
- Experiment folders with training scripts and saved results (base kaggle code found here: https://www.kaggle.com/code/maryammather/neural-network):
    - `first_version/` 
    - `extra_hidden_layer/`
    - `no_dropout/`
    - `optimal/`
    - `weighted_loss/` (uses weighted loss to address class imbalance)
- `mgenre.py` — a Kaggle-notebook-style end-to-end script for preprocessing, training baseline models, and producing a submission file. (can also be found here: https://www.kaggle.com/code/mandikudzadangwa/mgenre-py)


Overview

This repository contains several PyTorch-based experiments for music genre classification.
The full dataset is not included in this submission. To reproduce results you must run the provided notebook/scripts directly on Kaggle.

Kaggle Notebook:

    https://www.kaggle.com/code/maryammather/neural-network
    

You can also download the dataset locally using the Kaggle CLI (recommended) or by using the Kaggle Notebook 'Add data' UI and run the dataset with this code.

Using the Kaggle CLI to download and unzip locally (example):

    kaggle datasets download -d <USERNAME>/<DATASET-NAME> -p dataset --unzip

After downloading, ensure the following pre-cleaned CSV files exist (these are the files the training scripts expect):

    dataset/cleaned_data/X_train_clean.csv
    dataset/cleaned_data/y_train_clean.csv
    dataset/cleaned_data/X_val_clean.csv
    dataset/cleaned_data/y_val_clean.csv
    dataset/cleaned_data/X_test_clean.csv
    dataset/cleaned_data/y_test_clean.csv

Requirements: 
Recommended Python: 3.8+ (works with 3.8, 3.9, 3.10). Install dependencies in a virtual environment.

Example (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Minimal dependencies are listed in `requirements.txt`.

How to run locally:

General notes:
- Scripts were written to run inside a Kaggle Notebook (so they may reference `/kaggle/input/<dataset-folder>`). If running locally, change the `data_dir` variable near the top of each script to `dataset/cleaned_data` or move the cleaned CSVs into a folder matching the scripts' paths.

Run the main notebook-style scripts (examples):

    # Run the Kaggle-notebook style script (mgenre.py)
    python mgenre.py

    # Run the weighted-loss PyTorch training script
    python weighted_loss\nn_weighted_loss.py

    # Other experiment scripts (each folder contains a script):
    python first_version\neural_network.py
    python extra_hidden_layer\nnn_extra_hidden_layer.py  # if named differently, open the file and run that script
    python no_dropout\nnn_no_dropout.py
    python optimal\nnn_optimal_version.py

Expected outputs:

- Trained model files (for example `genre_model.pth` saved by `nn_weighted_loss.py`).
- Training logs printed to stdout. 

How to run on Kaggle Notebooks (recommended for exact reproduction):

1. Create a new Kaggle Notebook (Kernel).
2. Click "Add data" and add the Kaggle dataset (the same dataset link you used above).
3. Upload repository code 
4. Make sure the Notebook's dataset path matches the script expectations. 
5. Run the notebook/script cells. Outputs will be viewable in the Notebook and saved to the session workspace.
