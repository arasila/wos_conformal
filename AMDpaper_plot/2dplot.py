import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

# Function to read the data from a text file
def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return x, y, z

percent = 0.01
c1 = "black"
c2 = "black"

# Define the colormaps to match the images as closely as possible
# cmap1 = LinearSegmentedColormap.from_list('custom_cmap1', ['#2a4858', '#8f705f', '#ffcc00'])
# cmap2 = LinearSegmentedColormap.from_list('custom_cmap2', ['#002255', '#d17c33', '#ffeb00'])
cmap1 = 'inferno'
cmap2 = 'inferno'

# Define judgment functions
def type_A(x, y):
    if np.sqrt(x * x + y * y) > 1:
        return False
    elif np.sqrt((x + 1.16452) ** 2 + (y + 0.482362) ** 2) < 0.767327:
        return False
    elif np.sqrt((x - 1.0) ** 2 + (y - 0.57735) ** 2) < 0.57735:
        return False
    return True

def type_B(x, y):
    if x * x + y * y > 1:
        return False
    return True

def naive_rectangle(x, y):
    if x < 0 or x > 1:
        return False
    elif y < 0 or y > 1:
        return False
    return True

def L_shape(x, y):
    if x < 0 or x > 3:
        return False
    elif y < 0 or y > 2:
        return False
    elif x > 2 and y > 1:
        return False
    return True

# Function to create contour plots
def create_contour_plots(x1, y1, z1, x2, y2, z2, color1, color2, func, name, levels=10, contour_linewidth=1.0, boundary_linewidth=2.0):
    # Create grid data for contour plot
    grid_x, grid_y = np.mgrid[min(min(x1), min(x2)):max(max(x1), max(x2)):500j,
                     min(min(y1), min(y2)):max(max(y1), max(y2)):500j]
    grid_z1 = griddata((x1, y1), z1, (grid_x, grid_y), method='cubic')
    grid_z2 = griddata((x2, y2), z2, (grid_x, grid_y), method='cubic')

    # Define plotting range
    x = np.linspace(-10, 10, 8000)  # Increase the number of points for higher resolution
    y = np.linspace(-10, 10, 8000)
    X, Y = np.meshgrid(x, y)

    # Compute region
    Z = np.vectorize(func)(X, Y)

    # Mask the areas outside the defined region
    mask = np.vectorize(func)(grid_x, grid_y)
    grid_z1 = np.ma.masked_where(~mask, grid_z1)
    grid_z2 = np.ma.masked_where(~mask, grid_z2)

    # Color contour plot for first dataset
    plt.figure()
    plt.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=boundary_linewidth)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.contourf(grid_x, grid_y, grid_z1, levels=levels, cmap=cmap1, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(grid_x, grid_y, grid_z1, levels=levels, colors=color1, linewidths=contour_linewidth)
    plt.gca().set_xlim(min(min(x1), min(x2)) - percent * (max(grid_x.flatten()) - min(min(x1), min(x2))),
                       max(grid_x.flatten()) + percent * (
                                   max(grid_x.flatten()) - min(min(x1), min(x2))))  # Expand x limits
    plt.gca().set_ylim(min(min(y1), min(y2)) - percent * (max(grid_y.flatten()) - min(min(y1), min(y2))),
                       max(grid_y.flatten()) + percent * (
                                   max(grid_y.flatten()) - min(min(y1), min(y2))))  # Expand y limits
    plt.gca().set_axisbelow(False)  # Ensure the border is on top of the axis
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(name + '_real.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # Color contour plot for second dataset
    plt.figure()
    plt.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=boundary_linewidth)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.contourf(grid_x, grid_y, grid_z2, levels=levels, cmap=cmap2, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(grid_x, grid_y, grid_z2, levels=levels, colors=color2, linewidths=contour_linewidth)
    plt.gca().set_xlim(min(min(x1), min(x2)) - percent * (max(grid_x.flatten()) - min(min(x1), min(x2))),
                       max(grid_x.flatten()) + percent * (
                                   max(grid_x.flatten()) - min(min(x1), min(x2))))  # Expand x limits
    plt.gca().set_ylim(min(min(y1), min(y2)) - percent * (max(grid_y.flatten()) - min(min(y1), min(y2))),
                       max(grid_y.flatten()) + percent * (
                                   max(grid_y.flatten()) - min(min(y1), min(y2))))  # Expand y limits
    plt.gca().set_axisbelow(False)  # Ensure the border is on top of the axis
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(name + '_imaginary.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # Contour lines plot with black contour lines and no colorbar for both datasets
    plt.figure()
    plt.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=boundary_linewidth)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.contour(grid_x, grid_y, grid_z1, levels=levels, colors=color1, linewidths=contour_linewidth)
    plt.contour(grid_x, grid_y, grid_z2, levels=levels, colors=color2, linewidths=contour_linewidth)

    plt.gca().set_xlim(min(min(x1), min(x2)) - percent * (max(grid_x.flatten()) - min(min(x1), min(x2))),
                       max(grid_x.flatten()) + percent * (
                                   max(grid_x.flatten()) - min(min(x1), min(x2))))  # Expand x limits
    plt.gca().set_ylim(min(min(y1), min(y2)) - percent * (max(grid_y.flatten()) - min(min(y1), min(y2))),
                       max(grid_y.flatten()) + percent * (
                                   max(grid_y.flatten()) - min(min(y1), min(y2))))  # Expand y limits
    plt.gca().set_axisbelow(False)  # Ensure the border is on top of the axis
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(name + '_combined.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

data_file = [['NaiveRectangle', naive_rectangle, 10],
             ['L_shaped_quadrilateral', L_shape, 14],
             ['circularArc_typeA', type_A, 10],
             ['circularArc_typeB', type_B, 10]]

for I in data_file:
    # Read data from the files
    x1, y1, z1 = read_data(I[0] + '_data.txt')
    x2, y2, z2 = read_data(I[0] + '_conjugate_data.txt')
    create_contour_plots(x1, y1, z1, x2, y2, z2, c1, c2, I[1], I[0], levels=I[2], contour_linewidth=1.0, boundary_linewidth=2.0)
