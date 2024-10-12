import numpy as np
import matplotlib.pyplot as plt
import json


# Plot scaling results of1024euler_distributed.py
# mcups = million cell updates per second
# time in seconds

with open("runs.json", "r") as file:
    runs = json.load(file)


def main():

    # Plot scaling results

    # plot macbook mcups vs resolution for jax and numpy
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
        mcups = [
            r["mcups"]
            for r in runs
            if r["computer"] == "macbook"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        ax.plot(n_cells, mcups, style, label=lib + " (" + prec + "-precision)")
    ax.set_xlabel("problem size: # cells")
    ax.set_ylabel("million cell updates per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([2e3, 2e6])
    ax.set_ylim([3e0, 1e2])
    ax.set_title("Macbook M3 Max - strong scaling")
    ax.legend()
    plt.show()
    fig.savefig("scaling_strong.png")

    # plot rusty mcups vs number of GPUs(/resolution) for jax
    fig, ax = plt.subplots()
    libraries = ["jax", "jax"]
    precisions = ["single", "double"]
    styles = ["gs-", "bs-"]
    for lib, prec, style in zip(libraries, precisions, styles):
        n_devices = [
            r["n_devices"]
            for r in runs
            if r["computer"] == "rusty"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        mcups = [
            r["mcups"]
            for r in runs
            if r["computer"] == "rusty"
            and r["library"] == lib
            and r["precision"] == prec
        ]
        ax.plot(n_devices, mcups, style, label=lib + " (" + prec + "-precision)")
    ax.set_xlabel("# gpus (# cells)")
    ax.set_ylabel("million cell updates per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.5, 32])
    ax.set_xticks([1, 4, 16])
    ax.set_xticklabels(
        [
            "1 (4096^2)",
            "4 (8192^2)",
            "16 (16384^2)",
        ]
    )
    ax.set_ylim([5e2, 3e4])
    ax.set_title("Rusty - weak scaling")
    ax.legend()
    plt.show()
    fig.savefig("scaling_weak.png")

    return


if __name__ == "__main__":
    main()
