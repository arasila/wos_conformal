import numpy as np
import matplotlib.pyplot as plt

# ================== 可调参数 ==================
GRID_N = 2000         # 网格精细度: 400 / 600 / 800 / 1000
N_LEVELS = 120        # 等高线密度，越大越细，可以到 150 或 200
# ===========================================

plt.rcParams["figure.dpi"] = 300

# 1. Load data
data = np.loadtxt("points.txt", delimiter=",")
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# 2. Create fine grid
xi = np.linspace(x.min(), x.max(), GRID_N)
yi = np.linspace(y.min(), y.max(), GRID_N)
X, Y = np.meshgrid(xi, yi)

# 3. Interpolate
try:
    from scipy.interpolate import griddata
    Z = griddata((x, y), z, (X, Y), method="cubic")
    nan_mask = np.isnan(Z)
    if np.any(nan_mask):
        Z_nn = griddata((x, y), z, (X, Y), method="nearest")
        Z[nan_mask] = Z_nn[nan_mask]
except:
    print("⚠️ SciPy not found — using nearest fallback")
    Z = np.zeros_like(X)
    for i in range(GRID_N):
        for j in range(GRID_N):
            dist = (x - xi[i])**2 + (y - yi[j])**2
            idx = np.argmin(dist)
            Z[j, i] = z[idx]

# 4. Mask out L-shape missing region
mask = (X > 2.0) & (Y > 1.0)
Z = np.ma.array(Z, mask=mask)

# 5. Levels
vmin, vmax = Z.min(), Z.max()
levels = np.linspace(vmin, vmax, N_LEVELS)

# 6. Plot
fig, ax = plt.subplots(figsize=(8, 4))

# Filled contour
cont_f = ax.contourf(
    X, Y, Z,
    levels=levels,
    cmap="viridis"
)

# Line contour (dense, no labels)
ax.contour(
    X, Y, Z,
    levels=levels,
    colors="black",
    linewidths=0.15
)

# Colorbar
cbar = plt.colorbar(cont_f, ax=ax)
cbar.set_label("u(x,y)")

# Draw boundary of L shape
ax.plot([2, 2], [1, 2], "k-", linewidth=1.2)
ax.plot([2, 3], [1, 1], "k-", linewidth=1.2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("L-shaped Domain — Dense Contour (No Labels)")
ax.set_aspect("equal", "box")

plt.tight_layout()
plt.show()