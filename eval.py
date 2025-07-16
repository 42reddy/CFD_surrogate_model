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


X_test = X[40:]
Y_test = Y[40:]

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


model = FNO_network(3, 2, 64, 32, 10, 0.3).to(device)
state_dict = torch.load("combined_fno_model.pth")
model.load_state_dict(state_dict)

with torch.no_grad():
    Y_pred = model(X_test_tensor)

from torch.nn.functional import mse_loss

mse = mse_loss(Y_pred, Y_test_tensor)
print(f"MSE on test set: {mse.item():.6f}")



fig, axes = plt.subplots(4, 3, figsize=(8, 10))  # 4 rows, 2 columns

for i in range(4):
    pred_u = Y_pred[-i, 0].cpu().numpy()
    true_u = Y_test_tensor[-i, 0].cpu().numpy()

    axes[i, 0].imshow(true_u, cmap='jet')
    axes[i, 0].set_title(f"True u (Sample {i})")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(pred_u, cmap='jet')
    axes[i, 1].set_title(f"Predicted u (Sample {i})")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(pred_u - true_u, cmap='jet')
    axes[i, 2].set_title(f"Error in u (Sample {i})")
    axes[i, 2].axis('off')


plt.tight_layout()
plt.show()





# check for physical consistency

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

def divergence_error(velocity):
    u, v = velocity[:, 0], velocity[:, 1]

    dudx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
    dvdy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0

    dudx = dudx[:, 1:-1, :]
    dvdy = dvdy[:, :, 1:-1]

    div = dudx + dvdy
    return torch.norm(div) / velocity.shape[0]

def vorticity_error(pred, true):
    u_p, v_p = pred[:, 0], pred[:, 1]
    u_t, v_t = true[:, 0], true[:, 1]

    dvdx_p = (v_p[:, :, 2:] - v_p[:, :, :-2]) / 2.0
    dudy_p = (u_p[:, 2:, :] - u_p[:, :-2, :]) / 2.0

    dvdx_t = (v_t[:, :, 2:] - v_t[:, :, :-2]) / 2.0
    dudy_t = (u_t[:, 2:, :] - u_t[:, :-2, :]) / 2.0

    # Match shapes: [B, H-2, W-2]
    vort_p = dvdx_p[:, 1:-1, :] - dudy_p[:, :, 1:-1]
    vort_t = dvdx_t[:, 1:-1, :] - dudy_t[:, :, 1:-1]

    return torch.norm(vort_p - vort_t) / torch.norm(vort_t)


with torch.no_grad():
    rel_l2 = relative_l2_error(Y_pred, Y_test_tensor)
    div_err = divergence_error(Y_pred)
    vort_err = vorticity_error(Y_pred, Y_test_tensor)

print(f"Relative L2 Error:     {rel_l2.item():.6f}")
print(f"Divergence Error:      {div_err.item():.6f}")
print(f"Vorticity Error:       {vort_err.item():.6f}")



