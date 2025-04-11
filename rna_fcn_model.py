import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load and preprocess data
def load_data(path='stanford-rna-3d-folding/train.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# One-hot encode sequences
def encode_sample(sequence, structure, loop_type, max_len=107):
    vocab = {
        'sequence': {'A': 0, 'C': 1, 'G': 2, 'U': 3},
        'structure': {'.': 0, '(': 1, ')': 2},
        'loop': {'E': 0, 'S': 1, 'H': 2, 'I': 3, 'B': 4, 'M': 5, 'X': 6}
    }

    seq_enc = np.zeros((max_len, len(vocab['sequence'])))
    struct_enc = np.zeros((max_len, len(vocab['structure'])))
    loop_enc = np.zeros((max_len, len(vocab['loop'])))

    for i in range(max_len):
        seq_enc[i, vocab['sequence'][sequence[i]]] = 1
        struct_enc[i, vocab['structure'][structure[i]]] = 1
        loop_enc[i, vocab['loop'][loop_type[i]]] = 1

    features = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)
    return features

# 2. FCN Model
class FCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FCN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # predict reactivity per position
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

# 3. Train function
def train(model, dataloader, loss_fn, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# 4. Main
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data()

    # Create features and labels
    X, y = [], []
    for entry in data:
        x = encode_sample(entry['sequence'], entry['structure'], entry['predicted_loop_type'])
        y_vals = np.array(entry['reactivity'])  # shape (107,)
        X.append(x)
        y.append(y_vals)

    X = np.array(X)  # (num_samples, 107, feature_dim)
    y = np.array(y)  # (num_samples, 107)

    num_samples, seq_len, feat_dim = X.shape

    # Flatten to feed into FCN per base
    X = X.reshape(-1, feat_dim)
    y = y.reshape(-1)

    # Remove NaNs (missing reactivity values)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    # Convert to PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Model
    model = FCN(input_dim=feat_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train(model, train_loader, loss_fn, optimizer, epochs=10)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).cpu().numpy()

    plt.scatter(y_val.numpy(), preds, alpha=0.3)
    plt.xlabel("True Reactivity")
    plt.ylabel("Predicted Reactivity")
    plt.title("FCN Reactivity Prediction")
    plt.grid(True)
    plt.show()
