import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sep_tpo_PDB_dataset import PDBDataset
from transformations import PointCloudTransform
from PhosNet_model import PointNetBinaryClassifier

# data
train_true_csv   = 'afternoon_trainset_pos_june_19.csv'
train_false_csv  = 'trainset_neg_june_18.csv'
test_true_csv    = 'afternoon_testset_pos_june_19.csv'
test_false_csv   = 'testset_neg_june_18.csv'

# ── Hyperparameters 
epochs         = 5
batch_size     = 1
learning_rate  = 1e-4

# ── Custom collate 
def custom_collate(batch):
    flat = []
    for s in batch:
        flat.extend(s if isinstance(s, list) else [s])
    coords_list, labels_list, fnames = [], [], []
    for s in flat:
        coords_list.append(torch.tensor(s['coordinates'], dtype=torch.float32))
        labels_list.append(torch.tensor(s['label'],       dtype=torch.float32))
        fnames.append(s['filename'])
    return {
        'coordinates': torch.stack(coords_list),
        'labels':      torch.stack(labels_list),
        'filename':    fnames
    }

def main():
    # 1) Load pre-split datasets
    transform     = PointCloudTransform()
    train_dataset = PDBDataset(train_true_csv,  train_false_csv, transform=transform)
    test_dataset  = PDBDataset(test_true_csv,   test_false_csv,  transform=transform)

    # 2) DataLoaders
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 4) Model, loss, optimizer
    model     = PointNetBinaryClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5) Tracking containers
    train_losses = []
    test_aucs     = []

    # 6) Epoch loop
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
            coords = batch['coordinates'].to(device)
            labels = batch['labels'].to(device)
            fnames = batch['filename']
            optimizer.zero_grad()
            try:
                outputs = model(coords)
            except Exception as e:
                print("Error during forward pass on these file(s):")
                for f in fnames:
                    print("   ", f)
                raise
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        # ——— EVALUATE ON TEST SET as validation ————————————————————————
        model.eval()
        all_out, all_lbl = [], []
        with torch.no_grad():
            for batch in test_loader:
                coords = batch['coordinates'].to(device)
                labels = batch['labels'].to(device)
                out    = model(coords)

                all_out.extend(out.cpu().numpy().ravel().tolist())
                all_lbl.extend(labels.cpu().numpy().ravel().tolist())

        epoch_auc = roc_auc_score(all_lbl, all_out)
        test_aucs.append(epoch_auc)
        print(f"Epoch {epoch}: Test AUC  = {epoch_auc:.4f}")

    # 7) Save per-epoch metrics
    pd.DataFrame({
        'epoch':      list(range(1, epochs+1)),
        'train_loss': train_losses,
        'test_auc':   test_aucs
    }).to_csv('epoch_metrics.csv', index=False)
    print("Saved per-epoch metrics to epoch_metrics.csv")

    # 8) Final evaluation on test set (again) to save detailed results & ROC
    model.eval()
    all_out, all_lbl, all_fnames = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            coords = batch['coordinates'].to(device)
            labels = batch['labels'].to(device)
            out    = model(coords)

            all_out.extend(out.cpu().numpy().ravel().tolist())
            all_lbl.extend(labels.cpu().numpy().ravel().tolist())
            all_fnames.extend(batch['filename'])

    final_auc = roc_auc_score(all_lbl, all_out)
    print(f"\nFinal Test Set AUC = {final_auc:.4f}")

    # 9) Save test results
    pd.DataFrame({
        'filename':   all_fnames,
        'true_label': all_lbl,
        'score':      all_out
    }).to_csv('test_results.csv', index=False)
    print("Saved test results to test_results.csv")

    # 10) Plot & save ROC curve for test set
    fpr, tpr, _ = roc_curve(all_lbl, all_out)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {final_auc:.3f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ROC curve plot to roc_curve.png")

    # 11) Save the final model
    torch.save(model, 'PhosNet_full.pth')
    print("Saved full model to PhosNet_full.pth")

if __name__ == '__main__':
    main()
