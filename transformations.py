import torch
import numpy as np
import random

class PointCloudTransform:
    def __init__(self, num_rotations=6):
        self.num_rotations = num_rotations

    def apply_rotation(self, coordinates):
        # random rotation angles
        theta_x = random.uniform(0, 2 * np.pi)
        theta_y = random.uniform(0, 2 * np.pi)
        theta_z = random.uniform(0, 2 * np.pi)

        # x-axis.
        rotation_x = torch.tensor([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ], dtype=torch.float32)

        # y-axis.
        rotation_y = torch.tensor([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ], dtype=torch.float32)

        # z-axis.
        rotation_z = torch.tensor([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))

        # Apply the rotation first 3
        coordinates[:, :3] = torch.matmul(coordinates[:, :3], rotation_matrix.T)
        return coordinates

    def __call__(self, sample):
        # Convert the coordinates into a PyTorch tensor.
        original_coords = torch.tensor(sample['coordinates'], dtype=torch.float32)
        coords_part = original_coords[:, :3]
        features_part = original_coords[:, 3:]

        augmented_samples = []

        # Append the original
        augmented_samples.append({
            'coordinates': np.array(original_coords),
            'label': sample['label'],
            'filename': sample['filename']
        })

        # Generate six random rotated versions
        for i in range(self.num_rotations):
            rotated_coords_part = self.apply_rotation(coords_part.clone())
            combined = torch.cat([rotated_coords_part, features_part], dim=1)
            augmented_samples.append({
                'coordinates': np.array(combined),
                'label': sample['label'],
                'filename': sample['filename']
            })

        # (original + 6 rotations)
        return augmented_samples
