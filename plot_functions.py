import numpy as np
import matplotlib.pyplot as plt
from energy_and_grad import get_e, grid_gen, get_path_e


def plot_contour(x_min, x_max, y_min, y_max, delta, opt_path, calc_type=1, atomic_symbols=None):
    """
    This function plots a contour map of the energy surface and overlays a path taken by an optimization algorithm.

    Parameters:
    x_min (float): minimum value of the x-axis
    x_max (float): maximum value of the x-axis
    y_min (float): minimum value of the y-axis
    y_max (float): maximum value of the y-axis
    delta (float): interval between points on the grid
    opt_path (list or array-like): an array or list containing the points visited by the optimization algorithm
    calc_type (int): the type of calculation to be performed (0 for quantum mechanical, 1 for classical)
    atomic_symbols (list): a list of atomic symbols corresponding to the atoms in the molecule

    Returns:
    None
    """
    if calc_type == 0:
        # Get the energies for all the points on the optimization path and plot the path
        all_e = get_path_e(opt_path, atomic_symbols=atomic_symbols, calc_type=calc_type)
        plot_path_e(all_e)
    elif calc_type == 1:
        # Generate a grid of points over the x and y range and calculate the energies at each point
        x, y = grid_gen(x_min, x_max, y_min, y_max, delta)
        z = get_e((x, y), calc_type=calc_type)

        # Set the contour levels for the plot
        type1 = np.linspace(np.min(z), np.max(z)/9, 25, endpoint=False)
        type2 = np.linspace(np.max(z)/9, np.max(z)/2, 3, endpoint=False)
        type3 = np.linspace(np.max(z)/2, np.max(z), 1, endpoint=False)
        contour_levels = np.hstack((type1, type2, type3))

        # Create the contour plot
        origin = 'lower'
        fig2, ax2 = plt.subplots(constrained_layout=True)
        cs3 = ax2.contourf(x, y, z, contour_levels, cmap='viridis', origin=origin, extend='both')
        cs3.cmap.set_under('yellow')
        cs3.cmap.set_over('cyan')
        fig2.colorbar(cs3)

        cs4 = ax2.contour(x, y, z, contour_levels, colors=('k',), linewidths=(1,), linestyles='solid', origin=origin)
        ax2.clabel(cs4, fmt='%2.1f', colors='w', fontsize=12)

        # Overlay the optimization path on the plot
        opt_path = np.asarray(opt_path)
        markers_on = np.arange(1, len(opt_path), 2)
        ax2.plot(opt_path[:, 0], opt_path[:, 1], '-gD', markevery=markers_on, markersize=2)

        # Show the plot
        plt.show()


def mol_projection(initial_geometry, final_geometry):
    # Compute the centroids
    initial_centroid = np.mean(initial_geometry, axis=0)
    final_centroid = np.mean(final_geometry, axis=0)

    # Subtract the centroids
    initial_geometry -= initial_centroid
    final_geometry -= final_centroid

    # Compute the covariance matrix
    covariance_matrix = np.cov(np.vstack([initial_geometry, final_geometry]).T)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvectors in descending order of their corresponding eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Take the first two eigenvectors as the basis for the 2D projection
    projection_matrix = eigenvectors[:, :2]

    # Project the initial and final geometries onto the 2D plane
    projected_initial_geometry = initial_geometry.dot(projection_matrix)
    projected_final_geometry = final_geometry.dot(projection_matrix)

    # Plot the 2D projections
    plt.plot(projected_initial_geometry[:, 0], projected_initial_geometry[:, 1], label='Initial')
    plt.plot(projected_final_geometry[:, 0], projected_final_geometry[:, 1], label='Final')
    plt.legend()
    plt.show()


def plot_path_e(path_e, x_label='Iterations', y_label='Energy (kcal/mol)',
                plot_title='Energy (kcal/mol) vs iterations'):
    """
    Plots the energy vs iterations graph for a given path.

    Args:
    path_e: list or numpy array containing the energy values for each iteration.
    x_label: string representing the label for x-axis. Default is 'Iterations'.
    y_label: string representing the label for y-axis. Default is 'Energy (kcal/mol)'.
    plot_title: string representing the title of the plot. Default is 'Energy (kcal/mol) vs iterations'.

    Returns:
    None
    """
    fig, axs = plt.subplots(2, constrained_layout=True)
    ax1 = axs[0]
    markers_on = np.arange(1, len(path_e), 2)
    path_e = path_e - np.min(path_e)
    ax1.plot(np.arange(len(path_e)), path_e, '-gD', markevery=markers_on, markersize=1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(plot_title)
    ax2 = axs[1]
    markers_on = np.arange(1, len(path_e), 2)
    path_e = path_e - np.min(path_e)
    ax2.plot(np.arange(len(path_e)), path_e, '-gD', markevery=markers_on, markersize=1)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_ylim([0, 10])
    ax2.set_title('Zoomed in version of the plot above')
    plt.show()
