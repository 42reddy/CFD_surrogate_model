import numpy as np
import matplotlib.pyplot as plt


def simulation(u, v, p, mask, inlet_velocity, nu, dx, dy, dt, N):
    dx2 = dx ** 2
    dy2 = dy ** 2
    denom = 2 * (dx2 + dy2)

    Ny, Nx = u.shape
    p_list = []

    for i in range(N):

        u_star = u[1:-1, 1:-1] + dt * (
            - u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            - v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
            + nu * ((u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx2 +
                    (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy2)
        )

        v_star = v[1:-1, 1:-1] + dt * (
            - u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
            - v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
            + nu * ((v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx2 +
                    (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy2)
        )

        u_star_full = u.copy()
        v_star_full = v.copy()
        u_star_full[1:-1, 1:-1] = u_star
        v_star_full[1:-1, 1:-1] = v_star

        rhs = ((u_star_full[1:-1, 2:] - u_star_full[1:-1, :-2]) / (2 * dx) +
               (v_star_full[2:, 1:-1] - v_star_full[:-2, 1:-1]) / (2 * dy)) / dt

        # Pressure Poisson solve
        p_i = p.copy()
        for _ in range(20):
            p_new = p_i.copy()
            p_new[1:-1, 1:-1] = (
                (p_i[1:-1, 2:] + p_i[1:-1, :-2]) * dy2 +
                (p_i[2:, 1:-1] + p_i[:-2, 1:-1]) * dx2 -
                rhs * dx2 * dy2
            ) / denom

            # Boundary conditions
            p_new[:, 0] = p_new[:, 1]     # Inlet: dp/dx = 0
            p_new[:, -1] = 0              # Outlet: p = 0
            p_new[0, :] = p_new[1, :]     # Bottom: dp/dy = 0
            p_new[-1, :] = p_new[-2, :]   # Top: dp/dy = 0
            p_new[mask == 1] = 0          # Obstacle

            p_i = p_new

        p = p_i
        p_list.append(p.copy())

        # Velocity correction
        u_star_full[1:-1, 1:-1] -= dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
        v_star_full[1:-1, 1:-1] -= dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

        # Velocity BCs
        u_star_full[:, 0] = inlet_velocity     # Inlet u
        v_star_full[:, 0] = 0                  # Inlet v
        u_star_full[:, -1] = u_star_full[:, -2]  # Outlet du/dx = 0
        v_star_full[:, -1] = v_star_full[:, -2]  # Outlet dv/dx = 0
        u_star_full[0, :] = 0
        u_star_full[-1, :] = 0
        v_star_full[0, :] = 0
        v_star_full[-1, :] = 0

        u_star_full[mask == 1] = 0
        v_star_full[mask == 1] = 0

        u = u_star_full
        v = v_star_full

    return u, v, p, p_list





