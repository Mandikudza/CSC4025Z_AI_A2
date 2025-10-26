# === Music Genre Classification — (Neural Network) ===

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# 0. Helper: find dataset folder under /kaggle/input
# ---------------------
input_root = '/kaggle/input'
dataset_dirs = [d for d in glob.glob(os.path.join(input_root, '*')) if os.path.isdir(d)]
print("Found dataset directories:", dataset_dirs)
if len(dataset_dirs) == 0:
    raise RuntimeError("No files found in /kaggle/input. Upload files (train.csv etc.) via +Add data -> Upload.")

data_dir = dataset_dirs[0]  # choose first dataset folder by default
print("Using data folder:", data_dir)


def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None

# Candidate filenames
candidates = { 'train': ['train.csv', 'train_X.csv', 'train_X.csv', 'train_X.csv'],
               'val': ['val.csv', 'val_X.csv'],
               'test': ['test.csv', 'test_X.csv', 'test.xls', 'test.xlsx', 'X_test.csv'],
               'submission_template': ['submission.xls', 'submission.xlsx', 'submission_template.xls', 'submission_template.xlsx'] }

found = {}
for key, names in candidates.items():
    found[key] = None
    for name in names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            # try both read_csv and read_excel
            if name.endswith('.xls') or name.endswith('.xlsx'):
                try:
                    found[key] = pd.read_excel(p)
                    print(f"Loaded {name} as {key} (excel).")
                    break
                except Exception:
                    continue
            else:
                df = try_read_csv(p)
                if df is not None:
                    found[key] = df
                    print(f"Loaded {name} as {key}.")
                    break

if found['train'] is None:
    # search for any CSV containing 'train' or the only CSV
    csvs = glob.glob(os.path.join(data_dir, '*.csv'))
    csvs += glob.glob(os.path.join(data_dir, '*.xls')) + glob.glob(os.path.join(data_dir, '*.xlsx'))
    if len(csvs) == 0:
        raise RuntimeError("No CSV/XLS files were found in the dataset folder.")
    # Try to pick train.csv first
    possible_train = None
    for p in csvs:
        if 'train' in os.path.basename(p).lower():
            possible_train = p
            break
    if possible_train is None:
        possible_train = csvs[0]
    # attempt to read it
    if possible_train.endswith('.xls') or possible_train.endswith('.xlsx'):
        df_all = pd.read_excel(possible_train)
    else:
        df_all = pd.read_csv(possible_train)
    print("Using single file and splitting into train/val/test:", os.path.basename(possible_train))
    # Expect label column named 'Class' or 'class'
    if 'Class' not in df_all.columns and 'class' in df_all.columns:
        df_all.rename(columns={'class':'Class'}, inplace=True)
    # If label column not present, raise
    if 'Class' not in df_all.columns:
        raise RuntimeError("Label column 'Class' not found in the dataset. Ask Person A to include it or rename it to 'Class'.")

    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Class' in numeric_cols:
        numeric_cols.remove('Class')
    print("Auto-detected numeric features:", numeric_cols)
    # simple impute and scale later; split now
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df_all, test_size=0.2, stratify=df_all['Class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Class'], random_state=42)
    found['train'] = train_df
    found['val'] = val_df
    found['test'] = test_df

# If pre-split provided, but val missing, split train into train/val
if found['val'] is None and found['train'] is not None:
    print("No validation set found; splitting train into train/val (80/20).")
    from sklearn.model_selection import train_test_split
    if 'Class' not in found['train'].columns and 'class' in found['train'].columns:
        found['train'].rename(columns={'class':'Class'}, inplace=True)
    train_df, val_df = train_test_split(found['train'], test_size=0.2, stratify=found['train']['Class'], random_state=42)
    found['train'] = train_df
    found['val'] = val_df

# If test is missing, use val as test (best effort)
if found['test'] is None:
    print("No test set found; using validation as test set (you should replace with a proper test split).")
    found['test'] = found['val']

# Quick sanity check
for k in ['train','val','test']:
    if found[k] is None:
        raise RuntimeError(f"Failed to obtain {k} dataset.")
    print(f"{k} shape:", found[k].shape)


# We will automatically detect numeric features used for modelling (exclude 'Class' and obvious text fields)
def select_features(df):
    # Prefer the same features Person A used if available
    preferred = ['Popularity','danceability','energy','key','loudness','mode','speechiness',
                 'acousticness','instrumentalness','liveness','valence','tempo','duration_in min/ms','time_signature']
    cols = [c for c in preferred if c in df.columns]
    if len(cols) > 0:
        return cols
    # otherwise fall back to all numeric columns except 'Class'
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != 'Class']
    return numeric

feature_cols = select_features(found['train'])
print("Selected feature columns:", feature_cols)

# Build X and y
X_train_df = found['train'][feature_cols].copy()
y_train_df = found['train']['Class'].copy()

X_val_df = found['val'][feature_cols].copy()
y_val_df = found['val']['Class'].copy()

X_test_df = found['test'][feature_cols].copy()
y_test_df = found['test']['Class'].copy()

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train_df)
X_val = imputer.transform(X_val_df)
X_test = imputer.transform(X_test_df)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_df)
y_val = le.transform(y_val_df)
y_test = le.transform(y_test_df)

num_classes = len(le.classes_)
input_size = X_train.shape[1]
print("Num classes:", num_classes, "Input dims:", input_size)
print("Label classes:", list(le.classes_))

# PyTorch Dataset and DataLoaders
batch_size = 64

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),   torch.tensor(y_val, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),  torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Define Model (MLP)
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

# Training setup
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


# Plot learning curves
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


# Final evaluation on test set
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
print("\nClassification Report:\n", classification_report(all_true, all_preds, target_names=le.classes_))

# Confusion matrix (nice plotting)
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

# ---------------------
# Save model and predictions
# ---------------------
out_model_path = 'genre_model.pth'
torch.save(model.state_dict(), out_model_path)
print("Saved model to", out_model_path)

# If a submission template exists, generate predictions for that file if provided
if found.get('submission_template') is not None:
    sub_template = found['submission_template'].copy()
    try:
        # assume template columns correspond to classes (one-hot columns) or ids
        if 'test' in found and found['test'] is not None:
            # use X_test (already loaded)
            pred_rows = []
            model.eval()
            with torch.no_grad():
                for Xb in DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=64):
                    Xb = Xb[0].to(device)
                    out = model(Xb)
                    preds = out.argmax(1).cpu().numpy()
                    pred_rows.extend(preds)
            # create a simple submission: one column 'predicted_class'
            submission_out = pd.DataFrame({'predicted_class': le.inverse_transform(pred_rows)})
            submission_out.to_excel('submission_predictions.xlsx', index=False)
            print("Saved submission_predictions.xlsx")
    except Exception as e:
        print("Could not create submission file automatically:", e)

print("Done.")
