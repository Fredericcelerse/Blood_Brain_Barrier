# MolGraphBBB
This is a small tutorial on how we can use AI and Computational Chemistry to predict the ability of a small molecule to pass (or not) the Blood Brain Barrier.

In this example, we developed a small Message Passing Neural Network that considered each molecule of the dataset as a graph, and learns using a vector made of atomic and bond properties. 

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

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
conda install -c conda-forge rdkit pandas
```
```
conda install -c pytorch pytorch
```
```
conda install -c pyg pyg pytorch-scatter
```

### Database

The database used in this small project is named as B3DB comes from the article of Meng et al. (https://www.nature.com/articles/s41597-021-01069-5, github repository: https://github.com/theochem/B3DB). We took only the first 1058 molecules in our example, but this number should be increased for production way in the future.  

## Code architecture

The code is structured like this:

### 1. Load the dataset
The dataset is loaded using the pandas library and the SMILESDataSet class we defined in the code. This class aims at convert each SMILES into a graph, using the "atom_features" and "bond_features" functions we defined by using the rdkit pre-implemented functions. At the end, a graph is generated and stored for each of the molecule loaded from the BBB.csv file. 

### 2. Split the data
As commonly done, the data are split randomly: 80% will be used for the training while 10% for the validation and 10% for the test. 

### 3. Initialize the model
The model is initialized by defining our Message Passing Neural Network (MPNN) architecture. This part is one of the critical steps when building a ML model as many hyperparameters can be optimized here. This will be explored in a future project on how to optimize hyperparameters.

### 4. Train the model
The model is then trained using the "run_epoch" function and the performance is printed at the end. 

### 5. Evaluate the performance of the model
The model performance is finally evaluated on the test set, containing structures that were not used during the training and validation steps.

### 6. Save the model after training
The model is finally saved and can be used in future script for other predictions.

This architecture is standard and corresponds to classical ML algorithms.

## Executing the code

To execute the code, just enter the following command:
```
python MPNN.py
```

The progression of the learning will be printed for each epoch, with the Loss and Accuracy. At the end of the training, you can save the model through a my_bbb_mpnn_model name, which can be then load in another script using the same model architecture and the command "torch.load(my_bbb_mpnn_model)".

