# A simple example of solving the Euler equations with JAX on distributed systems
# Philip Mocz (2024)

import os
import jax

USE_CPU_ONLY = False  # True  # False

flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_force_host_platform_device_count=8"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_FLAGS"] = flags

jax.config.update("jax_enable_x64", True)  # double-precision
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import matplotlib.pyplot as plt
import time


@jax.jit
def get_conserved(rho, vx, vy, P, gamma, vol):
    """Calculate the conserved variables from the primitive variables"""

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)) * vol

    return Mass, Momx, Momy, Energy


@jax.jit
def get_primitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Calculate the primitive variable from the conserved variables"""

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1)

    return rho, vx, vy, P


@jax.jit
def get_gradient(f, dx):
    """Calculate the gradients of a field"""

    # (right - left) / 2dx
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dx)

    return f_dx, f_dy


@jax.jit
def extrapolate_to_face(f, f_dx, f_dy, dx):
    """Extrapolate the field from face centers to faces using gradients"""

    f_XL = f - f_dx * dx / 2
    f_XL = jnp.roll(f_XL, -1, axis=0)  # right/up roll
    f_XR = f + f_dx * dx / 2

    f_YL = f - f_dy * dx / 2
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2

    return f_XL, f_XR, f_YL, f_YR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply fluxes to conserved variables to update solution state"""

    F += -dt * dx * flux_F_X
    F += dt * dx * jnp.roll(flux_F_X, 1, axis=0)  # left/down roll
    F += -dt * dx * flux_F_Y
    F += dt * dx * jnp.roll(flux_F_Y, 1, axis=1)

    return F


@jax.jit
def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)

    P_star = (gamma - 1) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star

    # find wavespeeds
    C_L = jnp.sqrt(gamma * P_L / rho_L) + jnp.abs(vx_L)
    C_R = jnp.sqrt(gamma * P_R / rho_R) + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy


def update(Mass, Momx, Momy, Energy, vol, dx, gamma, courant_fac):
    """Take a simulation timestep"""

    # get Primitive variables
    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, gamma, vol)

    # get time step (CFL) = dx / max signal speed
    dt = courant_fac * jnp.min(
        dx / (jnp.sqrt(gamma * P / rho) + jnp.sqrt(vx**2 + vy**2))
    )

    # calculate gradients
    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    P_dx, P_dy = get_gradient(P, dx)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * P_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * P_dy)
    P_prime = P - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(rho_prime, rho_dx, rho_dy, dx)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx_prime, vx_dx, vx_dy, dx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy_prime, vy_dx, vy_dy, dx)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P_prime, P_dx, P_dy, dx)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)

    return Mass, Momx, Momy, Energy, dt, rho


def main():
    """Finite Volume simulation"""

    if not USE_CPU_ONLY:
        jax.distributed.initialize()
    n_devices = jax.device_count()
    mesh = Mesh(mesh_utils.create_device_mesh((n_devices, 1)), ("x", "y"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y"))

    if jax.process_index() == 0:
        for env_var in [
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NODELIST",
            "SLURM_STEP_NODELIST",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS",
        ]:
            print(f'{env_var}: {os.getenv(env_var,"")}')
        print("Total number of processes: ", jax.process_count())
        print("Total number of devices: ", jax.device_count())
        print("List of devices: ", jax.devices())
        print("Number of devices on this process: ", jax.local_device_count())

    # Simulation parameters
    N = 8192  # 8192 4096 2048 1024 # resolution
    boxsize = 1.0
    gamma = 5.0 / 3.0  # ideal gas gamma
    courant_fac = 0.4
    t_stop = 2.0
    save_freq = 0.1
    if USE_CPU_ONLY:
        save_animation_path = "output_euler_distributed"
    else:
        save_animation_path = (
            "/mnt/home/pmocz/ceph/jax-euler-benchmarks/output_euler_distributed"
        )

    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y = jnp.meshgrid(xlin, xlin, indexing="ij")

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)

    if jax.process_index() == 0:
        print("X:")
        jax.debug.visualize_array_sharding(X)

    # Generate Initial Conditions - opposite moving streams with perturbation
    w0 = 0.1
    sigma = 0.05 / jnp.sqrt(2.0)
    rho = 1.0 + (jnp.abs(Y - 0.5) < 0.25)
    vx = -0.5 + (jnp.abs(Y - 0.5) < 0.25)
    vy = (
        w0
        * jnp.sin(4 * jnp.pi * X)
        * (
            jnp.exp(-((Y - 0.25) ** 2) / (2 * sigma**2))
            + jnp.exp(-((Y - 0.75) ** 2) / (2 * sigma**2))
        )
    )
    P = 2.5 * jnp.ones(X.shape)
    P = jax.lax.with_sharding_constraint(P, sharding)

    if jax.process_index() == 0:
        print("rho:")
        jax.debug.visualize_array_sharding(rho)
        print("P:")
        jax.debug.visualize_array_sharding(P)

    # Get conserved variables
    Mass, Momx, Momy, Energy = get_conserved(rho, vx, vy, P, gamma, vol)

    if jax.process_index() == 0:
        print("Mass:")
        jax.debug.visualize_array_sharding(Mass)

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # Simulation Main Loop
    tic = time.time()
    t = 0
    output_counter = 0
    n_iter = 0
    save_freq = 0.05
    while t < t_stop:

        # Time step
        Mass, Momx, Momy, Energy, dt, rho = update(
            Mass, Momx, Momy, Energy, vol, dx, gamma, courant_fac
        )

        # determine if we should save the plot
        save_plot = False
        if t + dt > output_counter * save_freq:
            save_plot = True
            output_counter += 1

        # update time
        t += dt

        # update iteration counter
        n_iter += 1

        # save plot
        if save_plot:
            for d in range(jax.local_device_count()):
                plt.imsave(
                    save_animation_path
                    + "/dump_rho"
                    + str(output_counter).zfill(6)
                    + "_"
                    + str(jax.process_index()).zfill(2)
                    + "_"
                    + str(d).zfill(2)
                    + ".png",
                    jnp.rot90(rho.addressable_data(d)),
                    cmap="jet",
                    vmin=0.8,
                    vmax=2.2,
                )

            # Print progress
            if jax.process_index() == 0:
                print("[it=" + str(n_iter) + " t=" + "{:.6f}".format(t) + "]")
                print(
                    "  saved state "
                    + str(output_counter).zfill(6)
                    + " of "
                    + str(int(jnp.ceil(t_stop / save_freq)))
                )

                # Print million updates per second
                cell_updates = X.shape[0] * X.shape[1] * n_iter
                total_time = time.time() - tic
                mups = cell_updates / (1e6 * total_time)
                print("  million cell updates / second: ", mups)

    if jax.process_index() == 0:
        print("Total time: ", total_time)
        print("rho:")
        jax.debug.visualize_array_sharding(rho)

        import numpy as np

        # for each output time, stich together images from each of the processes
        for snap in range(1, output_counter + 1):
            images = []
            for p in range(jax.process_count()):
                for d in range(jax.local_device_count()):
                    images.append(
                        plt.imread(
                            save_animation_path
                            + "/dump_rho"
                            + str(snap).zfill(6)
                            + "_"
                            + str(p).zfill(2)
                            + "_"
                            + str(d).zfill(2)
                            + ".png"
                        )
                    )

            plt.imsave(
                save_animation_path + "/rho" + str(snap).zfill(6) + ".png",
                np.concatenate(images, axis=1),
                cmap="jet",
                vmin=0.8,
                vmax=2.2,
            )


if __name__ == "__main__":
    main()
