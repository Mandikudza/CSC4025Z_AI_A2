# === Music Genre Classification — (Neural Network) ===

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Step 1: Load the pre-cleaned data
# ---------------------
data_dir = "/kaggle/input/results"  

X_train = pd.read_csv(f"{data_dir}/X_train_clean.csv").to_numpy()
y_train = pd.read_csv(f"{data_dir}/y_train_clean.csv").squeeze().to_numpy()
X_val   = pd.read_csv(f"{data_dir}/X_val_clean.csv").to_numpy()
y_val   = pd.read_csv(f"{data_dir}/y_val_clean.csv").squeeze().to_numpy()
X_test  = pd.read_csv(f"{data_dir}/X_test_clean.csv").to_numpy()
y_test  = pd.read_csv(f"{data_dir}/y_test_clean.csv").squeeze().to_numpy()

print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Get parameters from the loaded data
num_classes = len(np.unique(y_train))
input_size = X_train.shape[1]
print("Num classes:", num_classes, "Input dims:", input_size)

# ---------------------
# Step 2: PyTorch Dataset and DataLoaders
# ---------------------
batch_size = 64

# Convert numpy arrays to tensors.
# Note: We use .long() for the 'y' tensors (labels)
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),   torch.tensor(y_val, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),  torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ---------------------
# Step 3: Define Model (MLP)
# ---------------------
class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=10, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenreClassifier(input_size=input_size, hidden_size=128, num_classes=num_classes, dropout_p=0.3).to(device)
print(model)

# ---------------------
# Step 4: Training setup
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 25

# Keep best model by val accuracy
best_val_acc = 0.0
best_state = None

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(1, epochs+1):
    # train
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    train_loss = total_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # validate
    model.eval()
    val_total_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            val_total_loss += loss.item() * Xb.size(0)
            preds = out.argmax(1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
    val_loss = val_total_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    print(f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.3f} | Val Loss {val_loss:.4f} Acc {val_acc:.3f}")

# restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    print("Loaded best model with Val Acc:", best_val_acc)

# ---------------------
# Step 5: Plot learning curves
# ---------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(train_accs, label='train acc')
plt.plot(val_accs, label='val acc')
plt.legend(); plt.title('Accuracy')
plt.show()

# ---------------------
# Step 6: Final evaluation on test set
# ---------------------
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

print("Test Accuracy:", accuracy_score(all_true, all_preds))
print("Test F1 (macro):", f1_score(all_true, all_preds, average='macro'))

class_names = [str(i) for i in range(num_classes)]
print("\nClassification Report:\n", classification_report(all_true, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

# ---------------------
# Step 7: Save model
# ---------------------
out_model_path = 'genre_model.pth'
torch.save(model.state_dict(), out_model_path)
print("Saved model to", out_model_path)

print("Done.")
