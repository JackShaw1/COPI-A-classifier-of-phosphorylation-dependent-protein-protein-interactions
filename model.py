import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointNetBinaryClassifier(nn.Module):
    def __init__(self, num_aa_types=20, num_atom_types=4, num_chain_ids=2):
        super(PointNetBinaryClassifier, self).__init__()

        self.num_atom_types = num_atom_types
        self.num_chain_ids = num_chain_ids
        self.aa_embedding = nn.Embedding(num_aa_types + 1, 3)
        self.pae_embedding = nn.Linear(1, 3)
        self.plddt_embedding = nn.Linear(1, 1)

        in_channels = 3 + 3 + (num_chain_ids + 1) + (num_atom_types + 1) + 3 + 1

        self.conv1 = nn.Conv1d(in_channels, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)
        self.conv3 = nn.Conv1d(32, 64, 1)

        self.bn1   = nn.InstanceNorm1d(16, affine=True)
        self.bn2   = nn.InstanceNorm1d(32, affine=True)
        self.bn3   = nn.InstanceNorm1d(64, affine=True)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, num_points, _ = x.shape

        # Extract spatial coordinates
        xyz = x[:, :, :3].permute(0, 2, 1)

        # One-hot encode the chain and atom ID
        chain_onehot = F.one_hot(x[:, :, 3].long(), num_classes=self.num_chain_ids + 1).float()
        chain_onehot = chain_onehot.permute(0, 2, 1)
        atom_onehot = F.one_hot(x[:, :, 5].long(), num_classes=self.num_atom_types + 1).float()
        atom_onehot = atom_onehot.permute(0, 2, 1)

        # Embed the features
        aa_embed = self.aa_embedding(x[:, :, 4].long()).permute(0, 2, 1)
        pae_input = x[:, :, 6:7]  # (batch, points, 1)
        pae_embed = self.pae_embedding(pae_input)
        pae_embed = pae_embed.permute(0, 2, 1)  # shape: (batch, 3, points)
        plddt_input = x[:, :, 7:8]  # shape: (batch, points, 1)
        plddt_embed = self.plddt_embedding(plddt_input)
        plddt_embed = plddt_embed.permute(0, 2, 1)  # shape: (batch, 1, points)

        # xyz (3) + aa_embed (3) + chain_onehot (3) + atom_onehot (5) + pae_embed (3) + plddt_embed (1) = 18 channels.
        features = torch.cat([xyz, aa_embed, chain_onehot, atom_onehot, pae_embed, plddt_embed], dim=1)

        # Apply convolutions
        features = self.conv1(features)
        features = self.bn1(features)
        features = F.relu(features)

        features = self.conv2(features)
        features = self.bn2(features)
        features = F.relu(features)

        features = self.conv3(features)
        features = self.bn3(features)

        # Global max pooling
        features = torch.max(features, 2, keepdim=False)[0]

        # FC layers
        x = F.relu(self.fc1(features))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)
