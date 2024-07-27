import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Function to read the data from a text file
def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    m = data[:, 3]
    return x, y, z, m

percent = 0.01
c1 = "black"
c2 = "black"

# Define the inside function
def is_inside(x, y, z):
    if x < 0 or x > 3:
        return False
    elif y < 0 or y > 1:
        return False
    elif z < 0 or z > 2:
        return False
    elif x > 1 and z > 1:
        return False
    return True

# Function to create a 3D scatter plot with color representing the temperature
def create_3d_scatter_plots(data, inside_func, cmap, file_name):
    X, Y, Z, W = data

    # Define a finer grid for higher resolution
    grid_x, grid_y, grid_z = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j, Z.min():Z.max():100j]

    # Interpolate the data
    points = np.array([X, Y, Z]).T
    values = W
    grid_W = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')

    # Mask out values outside the defined region
    mask = np.vectorize(inside_func)(grid_x, grid_y, grid_z)
    grid_W = np.ma.array(grid_W, mask=~mask)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the grid arrays and the interpolated values
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    flat_z = grid_z.flatten()
    flat_w = grid_W.flatten()

    # Use the interpolated data to plot a scatter plot
    scatter = ax.scatter(flat_x, flat_y, flat_z, c=flat_w, cmap=cmap, marker='o', alpha=0.5)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])

    # Set ticks
    ax.set_xticks(np.arange(0, 3.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))  # Making the Y-axis ticks sparser
    ax.set_zticks(np.arange(0, 2.1, 0.5))

    # Remove the axes labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Set background color and disable the gray pane color
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Save the plot in PNG format with specified DPI
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')

# File paths to the text file containing the 3D points data
file_path1 = 'L_shaped_polyhedron_data.txt'

# Read data from the file
m1 = read_data(file_path1)

# Create the 3D scatter plots for the dataset with different colormaps
create_3d_scatter_plots(m1, is_inside, 'inferno', 'L_shaped_polyhedron_data_inferno.png')
create_3d_scatter_plots(m1, is_inside, 'jet', 'L_shaped_polyhedron_data_jet.png')

plt.show()
