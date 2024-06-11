#!/usr/bin/env python

from rdkit import Chem
import torch
from rdkit.Chem import SanitizeMol, SanitizeFlags
from torch_geometric.data import DataLoader,Data
from torch.utils.data import Dataset
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean
from torch import nn, optim
from sklearn.metrics import roc_auc_score
import random
import pandas as pd

def atom_features(atom):
    """Convert the atom's attributes into a feature vector."""
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumExplicitHs(),
        atom.GetIsAromatic(),
        atom.GetHybridization().real,
        atom.GetImplicitValence(),
        atom.GetMass() * 0.01,  # Normaliser la masse
        1 if atom.IsInRing() else 0
    ], dtype=torch.float)

def bond_features(bond):
    """Convert the bond's attributes into a feature vector."""
    bt = bond.GetBondType()
    return torch.tensor([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC
    ], dtype=torch.float)

def clean_and_convert_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to convert SMILES: {smiles}")
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        edge_index = []
        edge_attr = []
        for bond in bonds:
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [[start, end], [end, start]]
            edge_attr += [bond_features(bond), bond_features(bond)]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr, dim=0)
        x = torch.stack([atom_features(atom) for atom in atoms], dim=0)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        print(f"Error processing molecule {smiles}: {e}")
        return None

class SMILESDataSet(Dataset):
    def __init__(self, csv_file):
        super(SMILESDataSet, self).__init__()
        self.data_frame = pd.read_csv(csv_file)
        # Preparation of graphs and labels.
        self.graphs = []
        self.labels = []
        for index, row in self.data_frame.iterrows():
            graph = clean_and_convert_smiles(row['smiles'])
            if graph is not None:
                graph.y = torch.tensor([row['p_np']], dtype=torch.float) 
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_smiles(self):
        return self.smiles

def split_data(data_list, train_ratio=0.8, val_ratio=0.1):
    indices = list(range(len(data_list)))
    random.shuffle(indices)
    
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices

class SubsetData(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]

# Initialize the dataset
smiles_dataset = SMILESDataSet('BBBP.csv')

# Create a DataLoader for training
train_indices, val_indices, test_indices = split_data(smiles_dataset, train_ratio=0.8, val_ratio=0.1)

train_dataset = SubsetData(smiles_dataset, train_indices)
val_dataset = SubsetData(smiles_dataset, val_indices)
test_dataset = SubsetData(smiles_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# MPNN
class MPNN(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim):
        super(MPNN, self).__init__(aggr='add') 
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim + edge_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_input_dim + edge_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, edge_input_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if x.size(0) == 0:
            print("No nodes in the batch")
            return torch.zeros((batch.max().item() + 1, 1)).to(x.device)
        node_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = scatter_mean(node_out, batch, dim=0)
        #print("Output dimensions:", out.size())  
        return out


    def message(self, x_i, x_j, edge_attr, index, size_i):
        x_edge = torch.cat([x_j, edge_attr], dim=1)
        edge_attr = self.edge_mlp(x_edge)
        x_j = torch.cat([x_j, edge_attr], dim=1)
        return self.node_mlp(x_j)

# Initialization of the model and the optimizer
model = MPNN(node_input_dim=9, edge_input_dim=4)
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.BCEWithLogitsLoss()

# Training and evaluation loop
def run_epoch(loader, is_train=True):
    total_loss = 0
    y_true = []
    y_pred = []

    for data in loader:
        optimizer.zero_grad() if is_train else None
        out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        loss = loss_func(out, data.y.float().view(-1, 1))
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(out.detach().sigmoid().view(-1).tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc_score = roc_auc_score(y_true, y_pred)
    return avg_loss, auc_score

# Execute the training for a few epochs
for epoch in range(500):
    train_loss, train_auc = run_epoch(train_loader, is_train=True)
    val_loss, val_auc = run_epoch(val_loader, is_train=False)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}')

# After training, evaluate on the test set
test_loss, test_auc = run_epoch(test_loader, is_train=False)
print(f'Test Loss: {test_loss}, Test AUC: {test_auc}')

# Save the model after training
torch.save(model.state_dict(), 'my_bbb_mpnn_model.pth')
print("Model saved !")
