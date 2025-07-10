from simulation import simulation
import numpy as np


num_samples = 50
Nx, Ny = 256, 128
Lx, Ly = 2.0, 1.0
dx, dy = Lx / Nx, Ly / Ny

dt = 0.00005
T = 0.5
N = int(T / dt)


all_masks = []
all_vels = []
all_nus = []
all_u = []
all_v = []
all_p = []

for sample_id in range(num_samples):
    print(f"Simulating sample {sample_id+1}/{num_samples}")

    # params
    cx = np.random.uniform(0.4, 0.6)
    cy = np.random.uniform(0.4, 0.6)
    radius = np.random.uniform(0.05, 0.15)
    inlet_velocity = np.random.uniform(0.05, 0.2)
    nu = np.random.uniform(0.05, 0.2)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = (dist <= radius).astype(np.float32)

    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))

    u[mask == 1] = 0
    v[mask == 1] = 0
    p[mask == 1] = 0

    # Run simulation
    u_out, v_out, p_out, _ = simulation(u, v, p, mask, inlet_velocity, nu, dx, dy, dt, N)

    all_masks.append(mask)
    all_vels.append(inlet_velocity)
    all_nus.append(nu)
    all_u.append(u_out)
    all_v.append(v_out)
    all_p.append(p_out)


np.savez_compressed("cylinder_cfd_dataset.npz",
                    masks=np.stack(all_masks),
                    inlet_velocities=np.array(all_vels),
                    nus=np.array(all_nus),
                    u=np.stack(all_u),
                    v=np.stack(all_v),
                    p=np.stack(all_p))




