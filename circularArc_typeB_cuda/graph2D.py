import numpy as np
import matplotlib.pyplot as plt

# Load data: file format must be "x, y, value"
data = np.loadtxt("points.txt", delimiter=",")
x, y, val = data.T

plt.figure(figsize=(6,5))

# Scatter plot, value determines color
sc = plt.scatter(x, y, c=val, cmap="viridis", s=15)

plt.colorbar(sc, label="Value")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot of Computed Results")
plt.axis("equal")   # preserve geometry
plt.tight_layout()
plt.show()