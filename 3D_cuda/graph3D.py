import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def load_points(filename="points.txt"):

    data = np.loadtxt(filename, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    x, y, z, val = data.T
    return x, y, z, val

def set_equal_3d(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def plot_3d_scatter(x, y, z, val):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=val, cmap="viridis", s=30, edgecolor="none")

    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label("Result (probability / potential)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Walk-on-Spheres Result")

    set_equal_3d(ax)
    plt.tight_layout()
    plt.show()

def plot_slice_y(x, y, z, val, y0=0.5, tol=0.05):

    mask = np.abs(y - y0) < tol
    if not np.any(mask):
        print(f"in y ≈ {y0} no point, adjust y0 or tol")
        return

    xs = x[mask]
    zs = z[mask]
    vs = val[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xs, zs, c=vs, cmap="viridis", s=40)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Result")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Slice at y ≈ {y0} (tol={tol})")
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()

def main():
    x, y, z, val = load_points("points.txt")

    plot_3d_scatter(x, y, z, val)

    plot_slice_y(x, y, z, val, y0=0.5, tol=0.05)

if __name__ == "__main__":
    main()