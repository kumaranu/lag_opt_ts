import os.path
import numpy as np
import argparse
import pathlib
from lag_opt_lib.logger import get_logger
import os
import shutil


def read_in_one_xyz_file(file_path):
    """
    Reads in one .xyz file and returns the atomic symbols and coordinates.

    Args:
        file_path (str or pathlib.Path): The path to the .xyz file.

    Returns:
        tuple: A tuple containing:
            - atomic_symbols (list): A list of the atomic symbols of the
             atoms in the file.
            - coords (numpy.ndarray): A numpy array of shape (n_atoms, 3)
             containing the coordinates of the atoms in the file.
    """
    f = open(file_path, 'r')
    inf = f.readlines()
    n_atoms = int(inf[0].strip())
    atomic_symbols = [i.strip().split()[0] for i in inf[2:(n_atoms+2)]]
    coords = np.asarray([i.strip().split()[1:] for i in inf[2:]]).astype(float)
    f.close()
    return [atomic_symbols, coords]


def read_in_multiple_xyz_file(file_path,
                              n_images=10):
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


def read_in_multiple_xyz_file2(file_path):
    # Replace "/path/to/file.xyz" with the actual path to your .xyz file
    with open(file_path, "r") as f:
        # Read the number of atoms from the first line
        num_atoms = int(f.readline())
        atomic_symbols = []
        all_coords = []
        while True:
            # Read the comment line
            f.readline()

            # Try to read a configuration
            try:
                # Read the atomic positions
                data = np.genfromtxt(f, max_rows=num_atoms)
                # print('data:\n', data)
                # Extract the atomic symbols and positions
                symbols = data[:, 0]
                positions = data[:, 1:]
                atomic_symbols.append(symbols)
                all_coords.append(positions)
            except ValueError:
                # If we can't read a configuration, we've reached the end of the file
                break
    return atomic_symbols[-1], all_coords[-1]


def read_in():
    """
    Parse command-line arguments and return a namespace containing
    the parsed arguments.

    This function uses the argparse module to parse command-line
    arguments for a geometry optimization job or a transition state
    search using the geodesic algorithm and Lagrangian path
    optimization. The function returns a namespace containing
    the parsed arguments, which can be accessed as attributes
    of the namespace.

    Arguments:
    None.

    Returns:
    args (argparse.Namespace): A namespace containing the parsed
    arguments.

    Example:
    To parse command-line arguments and get the parsed arguments
    in a namespace, call the function as follows:

    ```
    args = read_in()
    ```
    """
    ps = argparse.ArgumentParser(
        description="Optimization and transition state finding package",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    ps.add_argument('--input_dir',
                    type=pathlib.Path,
                    default=str(os.getcwd()) + '/inputs',
                    help='Path to the directory containing input files.')
    ps.add_argument('--output_dir',
                    type=pathlib.Path,
                    default=str(os.getcwd()) + '/outputs',
                    help='Path to the directory containing the output files.')
    ps.add_argument('--calc_type',
                    type=int,
                    default=1,
                    help='This is variable to check the type of calculation'
                         'that is bring performed.'
                         'calc_type=1 analytic function: wolfe_schlegel function.'
                         'calc_type=0 molecular system.'
                         'Other cases harmonic function.')
    ps.add_argument('--job_type',
                    type=str,
                    default='opt',
                    help='job_type=opt means a geometry optimization job.'
                         'job_type=ts_lag means a transition state'
                         'calculation using lagrangian path optimization')
    ps.add_argument("--n_images",
                    type=int,
                    default=20,
                    help="Number of images.")
    ps.add_argument('--x_min',
                    type=float,
                    default=-2.0,
                    help='Lower bound for the x dimension.'
                         'It is used for contour plotting.')
    ps.add_argument('--x_max',
                    type=float,
                    default=2.0,
                    help='Upper bound for the x dimension.'
                         'It is used for contour plotting.')
    ps.add_argument('--y_min',
                    type=float,
                    default=-2.0,
                    help='Lower bound for the y dimension.'
                         'It is used for contour plotting.')
    ps.add_argument('--y_max',
                    type=float,
                    default=2.0,
                    help='Upper bound for the y dimension.'
                         'It is used for contour plotting.')
    ps.add_argument('--delta',
                    type=float,
                    default=0.07,
                    help='The spacing between the grid points'
                         'used to generate the contour plot.')
    ps.add_argument('--opt_type',
                    type=int,
                    default=0,
                    help='opt_type = 0 means locally built Steepest descent'
                         'algorithm will be called for geometry optimization.'
                         'opt_type = 1 means ase geometry optimizer will be'
                         'called for the geometry optimization.')
    ps.add_argument('--step_size_g_opt',
                    type=float,
                    default=0.0001,
                    help='Step size for the geometry optimization.')
    ps.add_argument('--step_size_l_opt',
                    type=float,
                    default=0.0001,
                    help='Step size for the Lagrangian path optimization.')
    ps.add_argument('--eps_g_opt',
                    type=float,
                    default=1e-9,
                    help='The epsilon value for the energy convergence'
                         'criteria used by the geometry optimization.')
    ps.add_argument('--convergence_type',
                    type=int,
                    default=1,
                    help='convergence_type = 0 means the full path will'
                         'be checked for convergence in the Lagrangian'
                         'path optimization and L2-norm of the path will'
                         'will be checked for the convergence criterion.'
                         'convergence_type = 1 means the energy of the'
                         'highest point along the path will be checked'
                         'for the Lagrangian path optimization criterion.')
    ps.add_argument('--re_meshing_type',
                    type=int,
                    default=2,
                    help='re_meshing_type = 0 means normal Lagrangian path'
                         'optimization will be performed.'
                         're_meshing_type = 1 means that the path will be altered'
                         'after each re_mesh_frequency-th Lagrangian optimization step'
                         'in such a way that two of the lowest energy points along the path'
                         '(other than the reactant and the product) will be removed and two'
                         'points will be added at near the highest point along the Lagrangian'
                         'path which are just the cartesian-mid-points of the neighboring'
                         'points to the immediate neighboring points.'
                         're_meshing_type = 2 means that in-addition to re_meshing as in re_meshing_type=1, the'
                         'spring force constants will be changed using a Gaussian weighted function'
                         'that enhances the force constants near the highest point along the'
                         'Lagrangian path but dampens the force constants near the edges of the path.'
                         'The range of this Gaussian will be (1 - change_factor, 1 + change_factor).')
    ps.add_argument('--re_mesh_frequency',
                    type=int,
                    default=1,
                    help='This is the frequency after the Lagrangian path will'
                         'be re_meshed during the path optimization process. For'
                         'example, re_mesh_frequency=2 means the path will be'
                         're_meshed after every two steps in the Lagrangian path'
                         'optimization.')
    ps.add_argument('--change_factor',
                    type=float,
                    default=0.1,
                    help='This is parameter to control the extent to which we change'
                         'the force constants when re_meshing_type=2 is used.'
                         'If change_factor=0.1 is used then a Gaussian function'
                         'is multiplied with the force constant array along the path'
                         'with range between (1 - change_factor, 1 + change_factor).')
    ps.add_argument('--eps_l_opt',
                    type=float,
                    default=1e-9,
                    help='The epsilon value for the energy convergence'
                         'criteria used by the Lagrangian path optimization.')
    ps.add_argument('--max_iter_g_opt',
                    type=int,
                    default=500,
                    help='Maximum number of iterations used for the'
                         'geometry optimization process.')
    ps.add_argument('--max_iter_l_opt',
                    type=int,
                    default=500,
                    help='Maximum number of iterations used for the'
                         'Lagrangian optimization process.')
    ps.add_argument('--minima1',
                    type=list,
                    default=[-1.1659786, 1.4766458],
                    help='This is one of the minimums.'
                         'It could also be the reactant geometry.')
    ps.add_argument('--minima2',
                    type=list,
                    default=[1.1330013, -1.4857527],
                    help='This is one of the other minimums.'
                         'It could also be the product geometry.')
    ps.add_argument('--initial_path_type',
                    type=int,
                    default=1,
                    help='initial_path_type = 0 means the initial path will go through'
                         'a special geometry whose cartesian coordinate is defined'
                         'using the argument called special_point below.'
                         'initial_path_type = 1 means the path will simply be a'
                         'straight cartesian line between the reactant and the product'
                         'geometries.')
    ps.add_argument('--special_point',
                    type=list,
                    default=[0.1, 0.1],
                    help='This is one of the points on the path.'
                         'It should connect the reactant geometry'
                         'to the product geometry.')
    ps.add_argument('--a',
                    type=float,
                    default=200.0,
                    help='The coefficient for the distance term'
                         'between the images.')
    ps.add_argument('--nth',
                    type=int,
                    default=10,
                    help='Frequency of the path energies to plot.'
                         'For example, nth=10 would mean that the'
                         'code will plot every 10th step in the'
                         'Lagrangian path optimization.')
    ps.add_argument('--geodesic_code',
                    type=str,
                    default='/home/kumaranu/Documents/testing_geodesic')
    ps.add_argument('--xyz_r_p',
                    type=pathlib.Path,
                    default='',
                    # default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_B.xyz',
                    # default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/B_C+D.xyz',
                    # default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_C+D.xyz',
                    help='This is a file that contains reactant and'
                         'product geometry in the xyz format.')
    ps.add_argument('--xyz_file_init_geom',
                    type=pathlib.Path,
                    default='/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A.xyz',
                    help='This is the starting geometry for the'
                         'geometry optimization')
    ps.add_argument('--action_type',
                    type=int,
                    default=1,
                    help='action_type = 0 means defining the displacement'
                         'term in action using exponential.'
                         'action_type = 1 means defining the displacement term in'
                         'action using just the distance between the two geometries.')
    ps.add_argument('--path_grad_type',
                    type=int,
                    default=1,
                    help='path_grad_type = 0 means the full gradient vector for'
                         'the path will be used in the Lagrangian path optimization.'
                         'path_grad_type = 1 means that only the component of the'
                         'Lagrangian gradient that is perpendicular to the path will'
                         'be used in the Lagrangian path optimization')
    ps.add_argument('--lag_opt_plot_type',
                    type=int,
                    default=1,
                    help='lag_opt_plot_type = 0 will plot the final optimized path only.'
                         'lag_opt_plot_type = 1 will plot the all the optimization'
                         'path after every frame_frequency-th frame in the Lagrangian'
                         'path optimization process in a gif file.')
    ps.add_argument('--frame_frequency',
                    type=int,
                    default=1,
                    help='frame_frequency is used in the gif file that shows the'
                         'Lagrangian path optimization process. For example, if'
                         'frame_frequency = 2, then every second path in the'
                         'optimization process will be plotted.')
    ps.add_argument('--frames_per_second',
                    type=int,
                    default=1,
                    help='This is parameter to control the speed of the Lagrangian'
                         'path optimization inside the gif file.')
    ps.add_argument('--model_software',
                    type=str,
                    default='mace_model',
                    help='The is the software used to get energy and forces.'
                         'model_software = mace_model uses a mace model'
                         'model_software = rdkit uses rdkit library')
    ps.add_argument('--ml_model_path',
                    type=pathlib.Path,
                    default='',
                    help='This is the path to the Mace model used for'
                         'energy and force calculations.')
    args = ps.parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    if args.calc_type == 0:
        atomic_symbols, x0 = read_in_one_xyz_file(args.xyz_file_init_geom)
        ps.add_argument('--init_geom',
                        type=list,
                        default=list(x0.flatten()),
                        help='Here, we update x0 with the initial geometry from'
                             'the xyz-formatted file provided by the user.')
        ps.add_argument('--atomic_symbols',
                        type=list,
                        default=atomic_symbols,
                        help='Here, we update x0 with the atomic labels from'
                             'the xyz-formatted file provided by the user.')
    elif args.calc_type == 1:
        ps.add_argument('--init_geom',
                        type=list,
                        default=[0.1, 0.1],
                        help='Initial point/geometry for the optimization job.')
        ps.add_argument('--atomic_symbols',
                        type=list,
                        default=[],
                        help='Here, no labels for analytic functions.')
    if args.job_type == 'ts_lag':
        ps.add_argument('--all_path_e_file',
                        type=pathlib.Path,
                        default=str(args.output_dir) + '/all_path_e.txt',
                        help='This is the file where the energies for the'
                             'Lagrangian paths will be saved.')

    args = ps.parse_args()
    logger = get_logger('my_logger',
                        log_file=str(args.output_dir) + '/lag_opt.log',
                        log_mode='w')
    logger.info('Inside lag-opt package.')
    logger.info('Read-in user arguments.')

    logger = get_logger('my_logger',
                        log_file=str(args.output_dir) + '/lag_opt.log')
    '''
    logger.debug(f"calc_type argument: {args.calc_type}")
    logger.debug(f"job_type argument: {args.job_type}")
    logger.debug(f"n_images argument: {args.n_images}")
    logger.debug(f"x_min argument: {args.x_min}")
    logger.debug(f"x_max argument: {args.x_max}")
    logger.debug(f"y_min argument: {args.y_min}")
    logger.debug(f"y_max argument: {args.y_max}")
    logger.debug(f"delta argument: {args.delta}")
    logger.debug(f"step_size_g_opt argument: {args.step_size_g_opt}")
    logger.debug(f"eps_g_opt argument: {args.eps_g_opt}")
    logger.debug(f"step_size_l_opt argument: {args.step_size_l_opt}")
    logger.debug(f"eps_l_opt argument: {args.eps_l_opt}")
    logger.debug(f"max_iter_g_opt argument: {args.max_iter_l_opt}")
    logger.debug(f"max_iter_l_opt argument: {args.max_iter_g_opt}")
    logger.debug(f"minima1 argument: {args.minima1}")
    logger.debug(f"minima2 argument: {args.minima2}")
    logger.debug(f"special_point argument: {args.special_point}")
    logger.debug(f"geodesic_code argument:"
                 f" {args.geodesic_code}")
    logger.debug(f"init_geom argument: {args.init_geom}")
    logger.debug(f"atomic_symbols argument: {args.atomic_symbols}")
    logger.debug(f"action_type argument: {args.action_type}")
    logger.debug(f"path_grad_type argument: {args.path_grad_type}")
    '''
    logger.debug(vars(args))
    return args
