import numpy as np
import matplotlib.pyplot as plt

# Plot scaling results of euler_distributed.py
# mups = million updates per second
# time in seconds

runs = [
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "double",
        "resolution": 64,
        "n_devices": 1,
        "mups": 4.6,
        "iterations": 874,
        "total_time": 0.7,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "double",
        "resolution": 128,
        "n_devices": 1,
        "mups": 15.0,
        "iterations": 1813,
        "total_time": 1.9,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "double",
        "resolution": 256,
        "n_devices": 1,
        "mups": 22.2,
        "iterations": 3694,
        "total_time": 10.9,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "double",
        "resolution": 512,
        "n_devices": 1,
        "mups": 27.0,
        "iterations": 7627,
        "total_time": 74.0,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "double",
        "resolution": 1024,
        "n_devices": 1,
        "mups": 36.2,
        "iterations": 15424,
        "total_time": 446.2,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "numpy",
        "precision": "double",
        "resolution": 64,
        "n_devices": 1,
        "mups": 7.4,
        "iterations": 874,
        "total_time": 0.4,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "single",
        "resolution": 64,
        "n_devices": 1,
        "mups": 4.1,
        "iterations": 874,
        "total_time": 0.8,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "single",
        "resolution": 128,
        "n_devices": 1,
        "mups": 17.9,
        "iterations": 1813,
        "total_time": 1.6,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "single",
        "resolution": 256,
        "n_devices": 1,
        "mups": 31.8,
        "iterations": 3694,
        "total_time": 7.5,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "single",
        "resolution": 512,
        "n_devices": 1,
        "mups": 53.6,
        "iterations": 7627,
        "total_time": 37.2,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "jax",
        "precision": "single",
        "resolution": 1024,
        "n_devices": 1,
        "mups": 57.8,
        "iterations": 15426,
        "total_time": 279.4,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "numpy",
        "precision": "double",
        "resolution": 128,
        "n_devices": 1,
        "mups": 10.8,
        "iterations": 1813,
        "total_time": 2.7,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "numpy",
        "precision": "double",
        "resolution": 256,
        "n_devices": 1,
        "mups": 12.0,
        "iterations": 3694,
        "total_time": 20.1,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "numpy",
        "precision": "double",
        "resolution": 512,
        "n_devices": 1,
        "mups": 10.2,
        "iterations": 7627,
        "total_time": 194.4,
    },
    {
        "computer": "macbook",
        "chip": "M3 Max",
        "library": "numpy",
        "precision": "double",
        "resolution": 1024,
        "n_devices": 1,
        "mups": 9.9,
        "iterations": 15424,
        "total_time": 1632.3,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "double",
        "resolution": 1024,
        "n_devices": 1,
        "mups": 199.3,
        "iterations": 15424,
        "total_time": 81.1,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "double",
        "resolution": 2048,
        "n_devices": 2,
        "mups": 621.5,
        "iterations": 32361,
        "total_time": 218.3,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "double",
        "resolution": 4096,
        "n_devices": 4,
        "mups": 1861.4,
        "iterations": 65277,
        "total_time": 588.3,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "double",
        "resolution": 8192,
        "n_devices": 8,
        "mups": 5265.9,
        "iterations": 134313,
        "total_time": 1711.6,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "single",
        "resolution": 1024,
        "n_devices": 1,
        "mups": 219.3,
        "iterations": 15427,
        "total_time": 73.7,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "single",
        "resolution": 2048,
        "n_devices": 2,
        "mups": 694.1,
        "iterations": 32986,
        "total_time": 199.3,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "single",
        "resolution": 4096,
        "n_devices": 4,
        "mups": 2579.1,
        "iterations": 66515,
        "total_time": 432.6,
    },
    {
        "computer": "rusty",
        "chip": "A100",
        "library": "jax",
        "precision": "single",
        "resolution": 8192,
        "n_devices": 8,
        "mups": 8719.6,
        "iterations": 137145,
        "total_time": 1055.5,
    },
]


def main():

    # Plot scaling results

    # plot macbook mups vs resolution for jax and numpy
    fig, ax = plt.subplots()
    libraries = ["jax", "jax", "numpy"]
    precisions = ["single", "double", "double"]
    styles = ["gs-", "bs-", "ro-"]
    for lib, prec, style in zip(libraries, precisions, styles):
        n_cells = [
            r["resolution"] ** 2
            for r in runs
            if r["computer"] == "macbook"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        mups = [
            r["mups"]
            for r in runs
            if r["computer"] == "macbook"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        ax.plot(n_cells, mups, style, label=lib + " (" + prec + "-precision)")
    ax.set_xlabel("problem size: # cells")
    ax.set_ylabel("million cell updates per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([2e3, 2e6])
    ax.set_ylim([3e0, 1e2])
    ax.set_title("Macbook M3 Max")
    ax.legend()
    plt.show()
    fig.savefig("scaling_strong.png")

    # plot rusty mups vs number of GPUs(/resolution) for jax
    fig, ax = plt.subplots()
    libraries = ["jax", "jax"]
    precisions = ["single", "double"]
    styles = ["gs-", "bs-"]
    for lib, prec, style in zip(libraries, precisions, styles):
        n_cells = [
            r["resolution"] ** 2
            for r in runs
            if r["computer"] == "rusty"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        mups = [
            r["mups"]
            for r in runs
            if r["computer"] == "rusty"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        ax.plot(n_cells, mups, style, label=lib + " (" + prec + "-precision)")
    ax.set_xlabel("problem size: # cells")
    ax.set_ylabel("million cell updates per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim([2e3, 2e6])
    # ax.set_ylim([3e0, 1e2])
    ax.set_title("Rusty")
    ax.legend()
    plt.show()
    fig.savefig("scaling_weak.png")

    return


if __name__ == "__main__":
    main()
