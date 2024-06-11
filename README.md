# Predicting the ability of small molecules to cross the Blood Brain Barrier
This is a small tutorial on how we can use AI and Computational Chemistry to predict the ability of a small molecule to pass (or not) the Blood Brain Barrier.

In this example, we developed a small Message Passing Neural Network that considered each molecule of the dataset as a graph, and learn using a vector made of atomic and bond properties. 

## Prerequists

### Anaconda

To execute the code, we will setup a specific environment using anaconda. To install it, go to this webpage: https://docs.anaconda.com/free/anaconda/install/

### Setup conda environment

We first create the conda environment: 
```
conda create -n bbb python=3.8
```

Then we activate the conda environment:
```
conda activate bbb
```

Once the environment is properly created, we will install the python libraries we need to execute the code:
```
conda install -c conda-forge rdkit==2024.03.2 torch==2.1.2.post104 pandas==2.0.3 torch-scatter==2.1.2 pyg==2.5.3
```

## Code architecture

The code is structured like this:

### 1. Load the dataset

### 2. Split the data

### 3. Initialize the model

### 4. Train the model

### 5. Evaluate the performance of the model

### 6. Save the model after training

## Exectuing the code

To execute the code, just enter the following command:
```
python MPNN.py
```

The progression of the learning will be printed for each epoch, with the Loss and Accuracy. At the end of the training, you can save the model through a my_bbb_mpnn_model name, which can be then load in another script using the same model architecture and the command "torch.load(my_bbb_mpnn_model)".

