import os
import csv
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, NeighborSearch
import re
import json


three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

residue_mapping = {
    "K": 0,
    "L": 1,
    "M": 2,
    "F": 3,
    "P": 4,
    "S": 5,
    "T": 6,
    "W": 7,
    "Y": 8,
    "V": 9,
    "A": 10,
    "R": 11,
    "N": 12,
    "D": 13,
    "C": 14,
    "Q": 15,
    "E": 16,
    "G": 17,
    "H": 18,
    "I": 19
}

atom_mapping = {
    'CA': 3, 'SD': 0, 'OG': 1, 'NE1': 2, 'OE1': 1, 'OE2': 1, 'OH': 1, 'NZ': 2, 'OG1': 1, 'OD1': 1, 'OD2': 1, 'NE2': 2, 'ND2': 2, 'NE': 2, 'NH1': 2, 'NH2': 2, 'SG': 0, 'ND1': 2
}

# Encode names for discrete biological features (aa name, atom name)
def encode_names(phos_id, bind_id, residue_name, atom_name):
    return (0 if phos_id == bind_id else 1), residue_mapping[three_to_one[residue_name]], atom_mapping[atom_name]

def extract_model_string(filename):
    match = re.search(r"(model_\d+)", filename)
    if match:
        return match.group(1)
    else:
        return ""

def get_chain_length(structure, chain_id):
    counter = 0
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                for residue in chain:
                    counter += 1
        break
    return counter

def get_pae_val(pae_data, phos_atom, bind_atom, chain_a_length):
    if phos_atom.get_parent().get_parent().get_id() == 'A' and bind_atom.get_parent().get_parent().get_id() == 'B':
        return pae_data[phos_atom.get_parent().get_id()[1] - 1][chain_a_length + bind_atom.get_parent().get_id()[1] - 1]
    if phos_atom.get_parent().get_parent().get_id() == 'B' and bind_atom.get_parent().get_parent().get_id() == 'A':
        return pae_data[chain_a_length + phos_atom.get_parent().get_id()[1] - 1][bind_atom.get_parent().get_id()[1] - 1]
    if phos_atom.get_parent().get_parent().get_id() == 'A' and bind_atom.get_parent().get_parent().get_id() == 'A':
        return pae_data[phos_atom.get_parent().get_id()[1] - 1][bind_atom.get_parent().get_id()[1] - 1]
    if phos_atom.get_parent().get_parent().get_id() == 'B' and bind_atom.get_parent().get_parent().get_id() == 'B':
        return pae_data[chain_a_length + phos_atom.get_parent().get_id()[1] - 1][chain_a_length + bind_atom.get_parent().get_id()[1] - 1]

def get_pae_dict(filename):
    folder, base_name = filename.split('/')[0], filename.split('/')[1]
    model_str = extract_model_string(base_name)
    for file in os.listdir(f'{folder}'):
        if model_str in file and base_name.split('aa_')[0] in file and '.json' in file:
            with open(f'{folder}/{file}', 'r') as file:
                pae_data = json.load(file)
                return pae_data['pae']

class PDBDataset(Dataset):
    def __init__(self, true_csv, false_csv, transform=None):
        self.samples = []
        self._load_csv(true_csv, label=1)
        self._load_csv(false_csv, label=0)
        self.transform = transform

    def _load_csv(self, csv_path, label):
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                filename, chain, residue_index = row[0], row[1], int(row[2])
                self.samples.append((filename, chain, residue_index, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, chain_id, residue_index, label = self.samples[idx]
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', f'{filename}')

        pae_data = get_pae_dict(filename)
        chain_a_length = get_chain_length(structure, 'A')

        point_cloud = []

        focal_atom = None
        model = structure[0]
        for chain in model:
            if chain.get_id() == chain_id:
                counter = 0
                for residue in chain:
                    if residue.get_resname() in three_to_one:
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                counter += 1
                    if counter == int(residue_index) and residue.get_resname() in three_to_one:
                        for atom in residue:
                            if residue.get_resname() == 'SER' and atom.get_name() == 'OG':
                                focal_atom = atom
                                chain_encoded, residue_name, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), residue.get_resname(), atom.get_name())
                                point_cloud.append(np.concatenate(([0, 0, 0], [chain_encoded, 20, atom_encoded, 0, atom.get_bfactor()])))
                            if residue.get_resname() == 'THR' and atom.get_name() == 'OG1':
                                focal_atom = atom
                                chain_encoded, residue_name, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), residue.get_resname(), atom.get_name())
                                point_cloud.append(np.concatenate(([0, 0, 0], [chain_encoded, 20, atom_encoded, 0, atom.get_bfactor()])))
                    #     for atom in residue:
                    #         if atom.get_name() == 'CA' and focal_atom != None:
                    #             coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                    #             chain_encoded, residue_name, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), residue.get_resname(), atom.get_name())
                    #             point_cloud.append(np.concatenate((coord, [chain_encoded, 20, atom_encoded, 0, atom.get_bfactor()])))
                    # if counter != int(residue_index) and abs(counter - int(residue_index)) < 3 and focal_atom != None: 
                    #     for atom in residue:
                    #         if atom.get_name() == 'CA':
                    #             res_name = atom.get_parent().get_resname()
                    #             atom_name = atom.get_name()
                    #             coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                    #             chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                    #             pae_val = get_pae_val(pae_data, focal_atom, atom, chain_a_length)
                    #             plddt = atom.get_bfactor()
                    #             point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded, pae_val, plddt])))

        if focal_atom is None:
            return None

        # Neighbors
        neighbor_search = NeighborSearch(list(structure.get_atoms()))
        neighbors_wide = neighbor_search.search(focal_atom.coord, 12)
        neighbors_narrow = neighbor_search.search(focal_atom.coord, 6)

        # Add atoms from the other chain
        for atom in neighbors_wide:
            if atom.get_parent().get_resname() in three_to_one and \
                atom.get_name() in atom_mapping and atom.get_name() != 'CA' \
                    and atom.get_parent().get_parent().get_id() != chain_id:
                        res_name = atom.get_parent().get_resname()
                        atom_name = atom.get_name()
                        coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                        chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                        pae_val = get_pae_val(pae_data, focal_atom, atom, chain_a_length)
                        plddt = atom.get_bfactor()
                        point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded, pae_val, plddt])))

        # Add atoms from the same chain
        for atom in neighbors_narrow:
            if atom.get_parent().get_resname() in three_to_one and \
                atom.get_name() in atom_mapping and atom.get_name() != 'CA' \
                    and atom.get_parent().get_parent().get_id() == chain_id \
                        and abs(atom.get_parent().get_id()[1] - focal_atom.get_parent().get_id()[1]) > 3:
                            res_name = atom.get_parent().get_resname()
                            atom_name = atom.get_name()
                            coord = (atom.coord - focal_atom.coord) / 12  # normalize coordinates
                            chain_encoded, res_encoded, atom_encoded = encode_names(chain.get_id(), atom.get_parent().get_parent().get_id(), res_name, atom_name)
                            pae_val = get_pae_val(pae_data, focal_atom, atom, chain_a_length)
                            plddt = atom.get_bfactor()
                            point_cloud.append(np.concatenate((coord, [chain_encoded, res_encoded, atom_encoded, pae_val, plddt])))

        if not point_cloud:
            return None

        sample = {'coordinates': np.array(point_cloud), 'label': label, 'filename': filename}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
