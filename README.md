# MolGraphBBB
This small project demonstrates how we can use AI and computational chemistry to predict the ability of a small molecule to cross the Blood-Brain Barrier (BBB).

In this example, we developed a Message Passing Neural Network (MPNN) that treats each molecule in the dataset as a graph, learning from a vector comprised of atomic and bond properties.

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

### Setup Conda Environment

First, create the conda environment:
```
conda create -n bbb python=3.8
```

Then, activate the conda environment:
```
conda activate bbb
```

Once the environment is properly set up, install the necessary Python libraries to execute the code:
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

The database used in this project, named BBBP, comes from the article by Martins et al. ([https://www.nature.com/articles/s41597-021-01069-5](https://pubmed.ncbi.nlm.nih.gov/22612593/), github repository: [https://github.com/theochem/B3DB](https://github.com/andresilvapimentel/bbbp-explainer)).

## Code Architecture

The code [MPNN.py](MPNN.py) is structured as follows:

### 1. Load the Dataset
The dataset is loaded using the pandas library and the SMILESDataSet class defined in our code. This class converts each SMILE into a graph using the atom_features and bond_features functions, implemented with RDKit's pre-existing functions. A graph is generated and stored for each molecule loaded from the BBB.csv file.

### 2. Split the Data
The data are randomly split: 80% for training, 10% for validation, and 10% for testing.

### 3. Initialize the Model
The model is initialized by defining our Message Passing Neural Network (MPNN) architecture. This step is critical as many hyperparameters can be optimized here, which will be explored in a future project on hyperparameter optimization.

### 4. Train the Model
The model is trained using the run_epoch function, and performance metrics are printed at the end.

### 5. Evaluate the Model Performance
The model's performance is evaluated on the test set, containing structures not used during the training or validation phases.

### 6. Save the Model After Training
The model is saved and can be used in future scripts for other predictions.

This architecture follows standard practices for ML algorithms.

## Executing the code

To run the code, enter the following command:
```
python MPNN.py
```

The progression of the training will be displayed for each epoch, including Loss and Accuracy. After training, you can save the model under the name [my_bbb_mpnn_model](my_bbb_mpnn_model), which can then be loaded in another script using the same model architecture with the command torch.load(my_bbb_mpnn_model).


