# Fault Classification in CAV Platoons
## Introduction
This repository contains the codes for my "Fault Classification in CAV Platoons" undergraduate thesis project. 

## Environment Setup
The `requirement.txt` file contains the dependencies required for the project. First, create a Python virtual environment with the command 

```
python -m venv <virtual_env_name>
```

This should create a new virtual environment with the name you specified. Then, activate this virtual environment by running its `activate` script. With the virtual environment activated, we can then install the requirements through pip using the command
```
pip install -r requirements.txt
```
This should install all necessary requirements. If PyTorch does not install, it is because pip is trying to install the CUDA compatible version. Please navigate to the official PyTorch website and follow the installation instructions given for CUDA compatibility, or edit the `requirements.txt` file by removing the `+cu116` from the `torch` dependency. 

## Usage
Various `.json` config files are used to select the model to train. These config files contain various directory paths that must be changed to match your local file structure. Some important notes about the expected directory layout:
- `train_data_dir`: Should contain all the data used for training/validation. The file path provided should link to a high level folder containing folders with the labels `drunk`, `delay`, `distracted`, `attack`, and `actuator` corresponding to the respective fault class. Within each fault class folder, the signal data must be in CSV format, with one row corresponding to one vehicle. 
- `test_data_dir`: Similar folder layout to `train_data_dir`, but in this case the folders should hold test data that will not be seen during network training. 
- `save_dir`: Folder in which all the training, testing, and visualization results will be stored.
- `model_dir`: Folder in which the trained models will be stored. Models are saved in ONNX format. 

Each config file also maintains a variety of configurable parameters that control the network architecture and the optimization process. 

The repository also provides a set of Python scripts for different aspects of the machine learning workflow. 
- `train_network.py` is the main training script. The script takes one of the `.json` configuration files as an additional command line argument to determine which type of model to train. 
- `test_network.py` is a model inference script that loads a saved ONNX model and evaluates its performance on test data. The script takes one of the `.json` configuration files as an additional command line argument to determine which model to test. It also takes an integer command line argument as the variance for the amount of noise to inject to the test data. To disable this additional noise, use the value `0` for this argument.  
- `train_svm.py` and `test_svm.py` provide training and testing scripts for an SVM model. These scripts are required since the SVM is not implemented in PyTorch.
- `train_all.py` and `test_all.py` are utility scripts for training and testing multiple models at once. 
- `visualize_data.py` provides a visualization of a set of random signals, one for each fault class. Data from both the training set and testing set are used.
- `sweep.py` integrates with the Weights and Biases API to set up hyperparameter sweep given the network config within the script. Note that this script is controlled by a hard-coded config object and not from any particular config file.