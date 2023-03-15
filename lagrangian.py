import numpy as np
import os
import readIn
from utils import standardization
from energy_and_grad import get_path_e

from logger import get_logger
logger = get_logger('my_logger', 'lag_opt.log')


def get_initial_path(calc_type, minima1, special_point, n_images, minima2,
                     path_to_a_script_to_call_geodesic_code=None, xyz_path=None,
                     xyz_r_p=None, atomic_symbols=None):
    """
    Generates the initial path for the optimization.

    Parameters:
    -----------
    calc_type : int
        The type of calculation to perform. 0 for geodesic path optimization, 1 for linear interpolation.
    minima1 : array_like
        The coordinates of the first minimum on the PES.
    special_point : array_like
        The coordinates of the special point on the PES.
    n_images : int
        The number of images to use for optimization.
    minima2 : array_like
        The coordinates of the second minimum on the PES.
    path_to_a_script_to_call_geodesic_code : str or None, optional
        The path to a script to call the geodesic code for calculation (only used if calc_type is 0).
        Default is None.
    xyz_r : str or None, optional
        The path to the reactant geometry file (only used if calc_type is 0). Default is None.
    xyz_p : str or None, optional
        The path to the product geometry file (only used if calc_type is 0). Default is None.
    xyz_path : str or None, optional
        The path to the geometry path file (only used if calc_type is 0). Default is None.
    xyz_r_p : str or None, optional
        The path to the combined reactant-product geometry file (only used if calc_type is 0). Default is None.
    atomic_symbols : list or None, optional
        The list of atomic symbols in the molecule. Default is None.

    Returns:
    --------
    list
        The list containing the reactant, product, path coordinates and atomic symbols.
    """
    if calc_type == 1:
        initial_points = np.vstack((minima1,
                                   np.linspace(minima1, special_point, n_images)[1:],
                                   np.linspace(special_point, minima2, n_images)[1:]))
        return [minima1, minima2, initial_points, atomic_symbols]
    elif calc_type == 0:
        '''
        This line right below should be converted to an in-memory call instead of system call.
        '''
        # Geodesic path optimization
        # Call the geodesic code
        reference_dir = os.getcwd()
        os.chdir(path_to_a_script_to_call_geodesic_code)
        os.system('geodesic_interpolate ' + str(xyz_r_p) + ' --output output.xyz --nimages ' + str(n_images))
        os.chdir(reference_dir)

        # Read in the reactant, product and path geometries
        atomic_symbols, coords_r_p = readIn.read_in_multiple_xyz_file(xyz_path, n_images=n_images)
        atomic_symbols_path, coords_path = readIn.read_in_multiple_xyz_file(
            path_to_a_script_to_call_geodesic_code + '/output.xyz', n_images=n_images)
        n_atoms = len(coords_r_p[0])
        coords_r = coords_r_p[0].flatten()
        coords_p = coords_r_p[1].flatten()
        coords_path = np.reshape(coords_path, (n_images, 3*n_atoms))
        # print('coords_path:', coords_path)
        return [coords_r, coords_p, coords_path, atomic_symbols]


def action(point_l, point_r, energy_l, energy_r, a, action_type=1):
    """
    Calculate the action variable for a given function f and two points point_l and point_r.

    Args:
    - f: a function that takes a point in n-dimensional space and returns a scalar value
    - point_l: a numpy array representing the coordinates of the left endpoint of the interval
    - point_r: a numpy array representing the coordinates of the right endpoint of the interval
    - a: a scalar value representing the strength of the distance component in the action variable

    Returns:
    - action_var: a scalar value representing the action variable for the given interval
    """
    dr = point_r - point_l
    df = energy_r - energy_l
    dist_component = 0
    if action_type == 0:
        dist_component = np.exp(a * (dr**2).sum()) - 1.0
    elif action_type == 1:
        dist_component = a * (dr**2).sum()
    potential_component = df ** 2
    action_var = dist_component + potential_component
    return action_var


def grad_action_l(point_l, point_r, energy_l, energy_r, grad_l, a, action_type=1):
    """
    Computes the gradient of the action with respect to the left point of the interval.

    Args:
        calc_type (int): Calculation type, determines the form of the potential energy function.
        atomic_symbols (list): List of atomic symbols.
        point_l (ndarray): Left point of the interval, shape (3 * n_atoms,).
        point_r (ndarray): Right point of the interval, shape (3 * n_atoms,).
        a (float): Exponential factor for the distance component of the action.

    Returns:
        action_grad_var (ndarray): Gradient of the action with respect to the left point, shape (3 * n_atoms,).
    """

    # Compute the difference between the right and left points
    d_r = point_r - point_l

    # Compute the difference in energy between the right and left points
    d_e = energy_r - energy_l
    dist_component_grad = np.zeros(len(d_r))
    if action_type == 0:
        # Compute the gradient of the distance component of the action
        dist_component_grad = -1 * np.exp(a * (d_r**2).sum()) * a * 2 * d_r
    elif action_type == 1:
        dist_component_grad = -1 * a * 2 * d_r

    # Compute the gradient of the graph component of the action
    graph_component_grad = -2 * d_e * grad_l

    # Compute the gradient of the action with respect to the left point
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r(point_l, point_r, energy_l, energy_r, grad_r, a, action_type=1):
    """
    Calculates the gradient of the action variable with respect to the right point in the interval.
    :param calc_type: An integer representing the type of calculation to be performed.
    :param atomic_symbols: A list of atomic symbols.
    :param point_l: A numpy array representing the left point in the interval.
    :param point_r: A numpy array representing the right point in the interval.
    :param a: A float representing the coefficient for the distance component of the action variable.
    :return: A numpy array representing the gradient of the action variable with respect to the right point.
    """
    d_r = (point_r - point_l)
    d_e = energy_r - energy_l
    dist_component_grad = np.zeros(len(d_r))
    if action_type == 0:
        dist_component_grad = np.exp(a * (d_r**2).sum()) * a * 2 * d_r
    elif action_type == 1:
        dist_component_grad = a * 2 * d_r
    graph_component_grad = 2 * d_e * grad_r
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def lagrangian(all_pts, all_e, a):
    """
    Compute the Lagrangian for a given function and set of points.

    Args:
    - f: the function to evaluate.
    - all_pts: a list of points that define the action.
    - a: a constant value.

    Returns:
    - The Lagrangian value computed for the function and set of points.

    """

    # Initialize the Lagrangian value to zero.
    lag = 0

    # Compute the action for each interval defined by the points.
    for i in range(len(all_pts)-1):

        # Add the action for the current interval to the Lagrangian value.
        lag += action(all_pts[i], all_pts[i+1], all_e[i], all_e[i+1], a)

    # Return the final Lagrangian value.
    return lag


def grad_lagrangian(all_pts, a, path_e, path_e_grad):
    """
    Compute the gradient of the Lagrangian for a given set of points.

    Args:
    - calc_type: a string indicating which calculation type to use.
    - atomic_symbols: a list of atomic symbols.
    - all_pts: a numpy array of points that define the action.
    - a: a constant value.

    Returns:
    - A numpy array representing the gradient of the Lagrangian.

    """
    # Create an array to hold the gradient of the Lagrangian.
    grad_lag = np.zeros((all_pts.shape[0]-2, all_pts.shape[1]))

    # Compute the gradient of the action for each interval defined by the points.
    for i in range(len(all_pts)-2):

        # Compute the gradient of the action for the left and right intervals and add them up.
        grad_lag[i] = np.array(grad_action_r(all_pts[i], all_pts[i+1], path_e[i], path_e[i+1],
                                             np.asarray(path_e_grad[i+1]).flatten(), a)) + \
                      np.array(grad_action_l(all_pts[i+1], all_pts[i+2], path_e[i+1], path_e[i+2],
                                             np.asarray(path_e_grad[i+1]).flatten(), a))

    # Return the computed gradient of the Lagrangian.
    return grad_lag


def find_critical_path(calc_type, atomic_symbols=None, initial_points=None,
                       start=None, end=None, num_steps=None, step_factor=None,
                       a=None, eps=0.001, all_path_e_file='all_path_e.txt',
                       output_path_file='output_path.txt'):
    """
    Find the critical path using gradient descent.

    Args:
    - calc_type: a number indicating which calculation type to use.
    - atomic_symbols: a list of atomic symbols.
    - initial_points: a numpy array of points.
    - start: a numpy array representing the start point.
    - end: a numpy array representing the end point.
    - num_steps: the number of steps to take.
    - step_factor: the factor by which to scale the step size.
    - a: a constant value.

    Returns:
    - A numpy array representing the points on the critical path.

    """
    # The following if conditions will go away soon.
    result, points = [], initial_points
    points_old = np.zeros(np.shape(points))
    e_r_p, e_grad_r_p = get_path_e(np.asarray([start, end]), atomic_symbols=atomic_symbols,
                                   calc_type=calc_type)
    start_e, start_e_grad, end_e, end_e_grad = e_r_p[0], e_grad_r_p[0], e_r_p[1], e_grad_r_p[1]
    all_path_e = []
    for step in range(num_steps):
        path_e, path_e_grad = get_path_e(points, atomic_symbols=atomic_symbols, calc_type=calc_type)

        path_e.insert(0, start_e)
        path_e.append(end_e)
        path_e_grad.insert(0, start_e_grad)
        path_e_grad.append(end_e_grad)
        all_path_e.append(path_e)
        all_pts = np.vstack((start, points, end))

        points = points - step_factor * grad_lagrangian(all_pts, a, path_e, path_e_grad)
        if np.linalg.norm(points_old - points) < eps:
            print('Lagrangian path optimization converged in ', step, 'steps.')
            break
        points_old = points

    # Save the final points to a file.
    np.savetxt(output_path_file, points, fmt='%.5f')
    np.savetxt(all_path_e_file, all_path_e, fmt='%.8f')
    print('Lagrangian path optimization converged in ', num_steps, 'steps.')

    # Return the final points.
    return np.vstack((start, points, end))
