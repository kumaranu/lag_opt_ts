import numpy as np
import argparse
import pathlib
from logger import get_logger


def read_in_one_xyz_file(file_path):
    """
    Reads in one .xyz file and returns the atomic symbols and coordinates.

    Args:
        file_path (str or pathlib.Path): The path to the .xyz file.

    Returns:
        tuple: A tuple containing:
            - atomic_symbols (list): A list of the atomic symbols of the atoms in the file.
            - coords (numpy.ndarray): A numpy array of shape (n_atoms, 3) containing the coordinates of
              the atoms in the file.
    """
    f = open(file_path, 'r')
    inf = f.readlines()
    n_atoms = int(inf[0].strip())
    atomic_symbols = [i.strip().split()[0] for i in inf[2:(n_atoms+2)]]
    coords = np.asarray([i.strip().split()[1:] for i in inf[2:]]).astype(float)
    f.close()
    return [atomic_symbols, coords]


def read_in_multiple_xyz_file(file_path, n_images=10):
    """
    Read in a specified number of XYZ files from a given file path.

    Parameters
    ----------
    file_path : str
        Path to the file containing XYZ files.
    n_images : int, optional
        Number of XYZ files to read in. Default is 10.

    Returns
    -------
    tuple
        A tuple containing:
            - A list of atomic symbols for each image.
            - A list of coordinates for each image.
    """
    f = open(file_path, 'r')
    inf = f.readlines()
    f.close()
    all_geoms = []
    n_atoms = int(inf[0].strip())
    atomic_symbols = [i.strip().split()[0] for i in inf[2:(n_atoms + 2)]]
    for i in range(n_images):
        coords = np.asarray([i.strip().split()[1:]
                             for i in inf[(i * (n_atoms + 2) + 2):(i+1)*(n_atoms + 2)]]).astype(float)
        all_geoms.append(coords)
    return [atomic_symbols, all_geoms]


def read_in():
    """
    Parse command-line arguments and return a namespace containing the parsed arguments.

    This function uses the argparse module to parse command-line arguments for a geometry optimization job or a
    transition state search using the geodesic algorithm and Lagrangian path optimization. The function returns
    a namespace containing the parsed arguments, which can be accessed as attributes of the namespace.

    Arguments:
    None.

    Returns:
    args (argparse.Namespace): A namespace containing the parsed arguments.

    Example:
    To parse command-line arguments and get the parsed arguments in a namespace, call the function as follows:

    ```
    args = read_in()
    ```
    """
    ps = argparse.ArgumentParser(description="Optimization and transition state finding package",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    ps.add_argument('--calc_type', type=int, default=1,
                    help='This is variable to check the type of calculation that is bring performed. '
                         'calc_type=1 means the analytic function using wolfe_schlegel function. '
                         'calc_type=0 means the molecular system. '
                         'Other cases such as harmonic functions should be added here, later.')
    ps.add_argument('--job_type', type=str, default='opt',
                    help='job_type=opt means a geometry optimization job.'
                         'job_type=ts_lag means a transition state calculation using lagrangian path'
                         'optimization')
    ps.add_argument("--n_images", type=int, default=20, help="Number of images.")
    ps.add_argument('--x_min', type=float, default=-2.0,
                    help='Lower bound for the x dimension. It is used for contour plotting.')
    ps.add_argument('--x_max', type=float, default=2.0,
                    help='Upper bound for the x dimension. It is used for contour plotting.')
    ps.add_argument('--y_min', type=float, default=-2.0,
                    help='Lower bound for the y dimension. It is used for contour plotting.')
    ps.add_argument('--y_max', type=float, default=2.0,
                    help='Upper bound for the y dimension. It is used for contour plotting.')
    ps.add_argument('--delta', type=float, default=0.07,
                    help='The spacing between the grid points used to generate the contour plot.')
    ps.add_argument('--step_size', type=float, default=0.0001,
                    help='Step size for the optimization.')
    ps.add_argument('--eps', type=float, default=1e-9,
                    help='The epsilon value for the energy convergence criteria used by the optimization process.')
    ps.add_argument('--max_iter', type=int, default=500,
                    help='Maximum number of iterations used for the optimization process.')
    ps.add_argument('--minima1', type=list, default=[-1.1659786, 1.4766458],
                    help='This is one of the minimums. It could also be the reactant geometry.')
    ps.add_argument('--minima2', type=list, default=[1.1330013, -1.4857527],
                    help='This is one of the other minimums. It could also be the product geometry.')
    ps.add_argument('--special_point', type=list, default=[0.1, 0.1],
                    help='This is one of the points on the path. It should connect the reactant '
                         'geometry to the product geometry.')
    ps.add_argument('--xyz_path', type=str, default='/home/kumaranu/Documents/testing_geodesic/inputs/12/output.xyz',
                    help='This is the path to an xyz-formatted file for the list of geometries defining an initial '
                         'path connecting the reactant to the product. It could be something like the output of the '
                         'geodesic code.')
    ps.add_argument('--a', type=float, default=200.0, help='The coefficient for the distance term between the images.')
    ps.add_argument('--nth', type=int, default=10, help='Frequency of the path energies to plot. '
                                                        'For example, nth=10 would mean that the code will plot every '
                                                        '10th step in the Lagrangian path optimization.')
    ps.add_argument('--path_to_a_script_to_call_geodesic_code', type=str,
                    default='/home/kumaranu/Documents/testing_geodesic')
    ps.add_argument('--all_path_e_file', type=pathlib.Path, default='all_path_e',
                    help='This is the file where the energies for the Lagrangian paths will be saved.')
    ps.add_argument('--xyz_r_p', type=pathlib.Path,
                    default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_C+D.xyz',
                    help='This is a file that contains reactant and product geometry in the xyz format.')
    ps.add_argument('--xyz_file_init_geom', type=pathlib.Path,
                    default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A.xyz',
                    help='This is the starting geometry for the geometry optimization')
    args = ps.parse_args()
    if args.calc_type == 0:
        atomic_symbols, x0 = read_in_one_xyz_file(args.xyz_file_init_geom)
        ps.add_argument('--init_geom', type=list, default=list(x0.flatten()),
                        help='Here, we update x0 with the initial geometry from the xyz-formatted file provided by '
                             'the user.')
        ps.add_argument('--atomic_symbols', type=list, default=atomic_symbols,
                        help='Here, we update x0 with the atomic labels from the xyz-formatted file provided by '
                             'the user.')
    elif args.calc_type == 1:
        ps.add_argument('--init_geom', type=list, default=[0.1, 0.1],
                        help='Initial point/geometry for the optimization job.')
        ps.add_argument('--atomic_symbols', type=list, default=[], help='Here, no labels for analytic functions.')
    args = ps.parse_args()

    logger = get_logger('my_logger', 'lag_opt.log')
    logger.debug(f"calc_type argument: {args.calc_type}")
    logger.debug(f"job_type argument: {args.job_type}")
    logger.debug(f"n_images argument: {args.n_images}")
    logger.debug(f"x_min argument: {args.x_min}")
    logger.debug(f"x_max argument: {args.x_max}")
    logger.debug(f"y_min argument: {args.y_min}")
    logger.debug(f"y_max argument: {args.y_max}")
    logger.debug(f"delta argument: {args.delta}")
    logger.debug(f"step_size argument: {args.step_size}")
    logger.debug(f"eps   argument: {args.eps  }")
    logger.debug(f"max_iter argument: {args.max_iter}")
    logger.debug(f"minima1 argument: {args.minima1}")
    logger.debug(f"minima2 argument: {args.minima2}")
    logger.debug(f"special_point argument: {args.special_point}")
    logger.debug(f"path_to_a_script_to_call_geodesic_code argument: {args.path_to_a_script_to_call_geodesic_code}")
    logger.debug(f"init_geom argument: {args.init_geom}")
    logger.debug(f"atomic_symbols argument: {args.atomic_symbols}")

    return args
