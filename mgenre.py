# This is a complete Kaggle Notebook script for music genre classification.
# You can copy-paste this into a Kaggle Notebook, upload the datasets (train.csv, test.xls, submission.xls),
# and run it. Make sure to enable GPU in the notebook settings for faster training.

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier  # For baseline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 2: Load the datasets
# Assuming you've uploaded train.csv, test.xls, and submission.xls to /kaggle/input/
train_df = pd.read_csv('/kaggle/input/your-dataset/train.csv')  # Replace 'your-dataset' with actual path if needed
test_df = pd.read_excel('/kaggle/input/your-dataset/test.xls')
submission_template = pd.read_excel('/kaggle/input/your-dataset/submission.xls')

# Print shapes and heads for verification
print("Train shape:", train_df.shape)
print(train_df.head())
print("Test shape:", test_df.shape)
print(test_df.head())

# Step 3: Preprocess the data
# Features to use (numerical only for simplicity; ignoring Artist Name and Track Name to avoid text processing)
features = [
    'Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_in min/ms',
    'time_signature'
]

# Handle mixed duration: Convert minutes to ms (assuming <1000 is minutes, else ms)
def normalize_duration(x):
    if x < 1000:  # Likely minutes
        return x * 60 * 1000
    return x

train_df['duration_in min/ms'] = train_df['duration_in min/ms'].apply(normalize_duration)
test_df['duration_in min/ms'] = test_df['duration_in min/ms'].apply(normalize_duration)

# Extract features and target
X = train_df[features]
y = train_df['Class']

# Handle missing values (e.g., empty strings in instrumentalness)
X = X.replace('', np.nan)
X = X.astype(float)  # Ensure all are float

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# For test set
X_test = test_df[features]
X_test = X_test.replace('', np.nan)
X_test = X_test.astype(float)
X_test = imputer.transform(X_test)  # Use same imputer

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split train into train/val/test (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test_internal, y_val, y_test_internal = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train/Val/Test split:", X_train.shape, X_val.shape, X_test_internal.shape)

# Ethical considerations: Music genre classification can perpetuate cultural stereotypes or biases in data 


# Step 4: Baseline - K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn_val = knn.predict(X_val)
y_pred_knn_test = knn.predict(X_test_internal)

print("Baseline KNN Validation Accuracy:", accuracy_score(y_val, y_pred_knn_val))
print("Baseline KNN Validation F1:", f1_score(y_val, y_pred_knn_val, average='macro'))
print("Baseline KNN Test Accuracy:", accuracy_score(y_test_internal, y_pred_knn_test))
print("Baseline KNN Test F1:", f1_score(y_test_internal, y_pred_knn_test, average='macro'))
print(classification_report(y_test_internal, y_pred_knn_test))

# Step 5: Neural Network in PyTorch
# Custom Dataset
class MusicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# DataLoaders
train_dataset = MusicDataset(X_train, y_train)
val_dataset = MusicDataset(X_val, y_val)
test_dataset = MusicDataset(X_test_internal, y_test_internal)
test_submission_dataset = MusicDataset(X_test_scaled)  # For final predictions

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
submission_loader = DataLoader(test_submission_dataset, batch_size=32, shuffle=False)

# Neural Network Architecture: Simple Feedforward NN
class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Hyperparameters (tune these as needed)
input_size = X_train.shape[1]  # 14 features
hidden_size = 128
num_classes = 11  # Classes 0-10
learning_rate = 0.001
epochs = 50

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenreClassifier(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Step 6: Evaluate NN on internal test set
model.eval()
y_pred_nn = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred_nn.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

print("NN Test Accuracy:", accuracy_score(y_true, y_pred_nn))
print("NN Test F1:", f1_score(y_true, y_pred_nn, average='macro'))
print(classification_report(y_true, y_pred_nn))

# Analysis: Compare to baseline. If NN underperforms, possible reasons: overfitting (check losses), 
# insufficient data, poor features (e.g., add text embeddings from Artist/Track if time), class imbalance.
# Tune hidden_size, epochs, add dropout, etc. Ethical: Model might misclassify niche genres due to data bias.

# Step 7: Generate predictions for submission (test set)
model.eval()
predictions = []
with torch.no_grad():
    for X_batch in submission_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Create submission DataFrame: One-hot encode predictions
submission_df = pd.DataFrame(0, index=range(len(predictions)), columns=submission_template.columns)
for i, pred in enumerate(predictions):
    submission_df.iloc[i, pred] = 1  # Set 1 in the predicted class column

# Save submission
submission_df.to_excel('submission_predictions.xls', index=False)
print("Submission saved as submission_predictions.xls")
