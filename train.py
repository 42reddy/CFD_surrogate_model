import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from FNO import  FNO_network
from tqdm import tqdm

data = np.load("cylinder_cfd_dataset.npz")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

masks = data["masks"]
inlet_vels = data["inlet_velocities"][:, None, None]
nus = data["nus"][:, None, None]
u = data["u"]
v = data["v"]

N, H, W = u.shape

X = np.stack([masks,
              np.tile(inlet_vels, (1, H, W)),
              np.tile(nus, (1, H, W))], axis=1)

Y = np.stack([u, v], axis=1)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

train_X, val_X = X_tensor[:40].to(device), X_tensor[40:].to(device=device)
train_Y, val_Y = Y_tensor[:40].to(device), Y_tensor[40:].to(device=device)

train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=4, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=4, shuffle=False)


model = FNO_network(3, 2, 64, 32, 10, 0.3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

n_epochs = 300


for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
    for xb, yb in train_bar:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False)
    with torch.no_grad():
        for xb, yb in val_bar:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


torch.save(model.state_dict(), "combined_fno_model.pth")
