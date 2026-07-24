import copy
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
MINORITY_KEEP_FRACTION = 0.04
def buildimbalancedsub(dataset):
    targets = dataset.targets.numpy()
    keep_indices = []
    rng = np.random.default_rng(SEED)
    for clas in range(10):
        clas_indices = np.where(targets == clas)[0]
        if clas == 8:
            n_keep = int(len(clas_indices) * 0.04)
            clas_indices = rng.choice(clas_indices, size=n_keep, replace=False)
        keep_indices.extend(clas_indices.tolist())
    return Subset(dataset, keep_indices)
transform = transforms.Compose([transforms.ToTensor()])
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_imbalanced = buildimbalancedsub(train_full)
train_loader = DataLoader(train_imbalanced, batch_size=128, shuffle=True)
test_loader = DataLoader(test_full, batch_size=256, shuffle=False)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = SmallCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train_one_epoch():
    model.train()
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def evaluate(eval_model):
    eval_model.eval()
    correct_per_class = np.zeros(10)
    total_per_class = np.zeros(10)
    confusion = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = eval_model(data)
            pred = output.argmax(dim=1)
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                total_per_class[t] += 1
                confusion[t, p] += 1
                if t == p:
                    correct_per_class[t] += 1

    per_class_acc = correct_per_class / total_per_class
    overall_acc = correct_per_class.sum() / total_per_class.sum()
    return overall_acc, per_class_acc, confusion
for epoch in range(6):
    train_one_epoch()
    acc, _, _ = evaluate(model)
    print(f"  Epoch {epoch + 1}/{6} - test accuracy: {acc:.4f}")

baseline_overall, baseline_per_class, _ = evaluate(model)
print(f"\nBaseline overall accuracy: {baseline_overall:.4f}")
print(f"Baseline per-class accuracy: {np.round(baseline_per_class, 4)}")

def compress_fc1_svd(source_model, rank):
    compressed = copy.deepcopy(source_model)
    W = compressed.fc1.weight.data.clone()   # shape (256, 1568)
    b = compressed.fc1.bias.data.clone()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    rank = min(rank, S.shape[0])
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]

    class FactorizedLinear(nn.Module):
        def __init__(self, U_k, S_k, Vt_k, bias):
            super().__init__()
            self.A = nn.Parameter(Vt_k.clone(), requires_grad=False)      
            self.B = nn.Parameter((U_k * S_k).clone(), requires_grad=False) 
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)

        def forward(self, x):
            return x @ self.A.T @ self.B.T + self.bias

    compressed.fc1 = FactorizedLinear(U_k, S_k, Vt_k, b).to(DEVICE)
    return compressed


def params_saved(rank, out_features=256, in_features=32 * 7 * 7):
    original = out_features * in_features
    factorized = rank * in_features + rank * out_features
    return 1 - (factorized / original)
ranks_to_test = [256, 128, 64, 32, 16, 8, 4, 2]

results = {
    "ranks": ranks_to_test,
    "compression_ratio_pct": [],
    "overall_accuracy": [],
    "minority_class_accuracy": [],      
    "majority_class_avg_accuracy": [], 
    "per_class_accuracy": [],
    "confusion_pairs": {"4_vs_9": [], "3_vs_5": [], "7_vs_1": []},
}

for rank in ranks_to_test:
    comp_model = compress_fc1_svd(model, rank)
    overall_acc, per_class_acc, confusion = evaluate(comp_model)

    pct_saved = params_saved(rank) * 100
    minority_acc = per_class_acc[8]
    majority_acc = np.mean([per_class_acc[i] for i in range(10) if i != 8])

    def confusion_rate(true_digit, confused_with):
        total = confusion[true_digit].sum()
        return confusion[true_digit, confused_with] / total if total > 0 else 0.0

    results["compression_ratio_pct"].append(round(pct_saved, 2))
    results["overall_accuracy"].append(round(float(overall_acc), 4))
    results["minority_class_accuracy"].append(round(float(minority_acc), 4))
    results["majority_class_avg_accuracy"].append(round(float(majority_acc), 4))
    results["per_class_accuracy"].append([round(float(x), 4) for x in per_class_acc])
    results["confusion_pairs"]["4_vs_9"].append(round(float(confusion_rate(4, 9)), 4))
    results["confusion_pairs"]["3_vs_5"].append(round(float(confusion_rate(3, 5)), 4))
    results["confusion_pairs"]["7_vs_1"].append(round(float(confusion_rate(7, 1)), 4))

    print(f"  rank={rank:>3} | params saved={pct_saved:6.1f}% | "
          f"overall={overall_acc:.4f} | minority({8})={minority_acc:.4f} | "
          f"majority_avg={majority_acc:.4f}")
with open("./results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("=" * 70)
