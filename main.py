import numpy as np
from lag_opt_lib import readIn
from lag_opt_lib.steepest_descent import steepest_descent
from lag_opt_lib import lagrangian
from lag_opt_lib.plot_functions import plot_contour, plot_multiple_path_es
from lag_opt_lib.logger import get_logger


# Check if the script is being run as the main program
if __name__ == '__main__':

    # Read input arguments from the command line
    args = readIn.read_in()

    logger = get_logger('my_logger', str(args.output_dir) + '/lag_opt.log')

    # If the job type is optimization, perform the steepest descent algorithm
    if args.job_type == 'opt':

        logger.info('This is an optimization job. Calling steepest descent function.')
        # Perform steepest descent optimization
        x, fx, num_iter, opt_path, path_e =\
            steepest_descent(args.init_geom,
                             args.step_size_g_opt,
                             args.eps_g_opt,
                             args.max_iter_g_opt,
                             calc_type=args.calc_type,
                             labels=args.atomic_symbols,
                             model_software=args.model_software,
                             output_dir=str(args.output_dir),
                             ml_model_path=str(args.ml_model_path))

        logger.info('Done with geometry optimization. Plotting the optimization path.')

        # Plot the optimization path
        plot_contour(x_min=args.x_min,
                     x_max=args.x_max,
                     y_min=args.y_min,
                     y_max=args.y_max,
                     delta=args.delta,
                     opt_path=opt_path,
                     path_e=path_e,
                     calc_type=args.calc_type)
    # If the job type is transition state optimization, perform Lagrangian path optimization
    elif args.job_type == 'ts_lag':
        logger.info('This is a transition state calculation.')
        logger.info('Generating an initial path for the Lagrangian path optimization.')

        # Obtain the initial path using geodesic code
        coords_r, coords_p, coords_path, atomic_symbols =\
            lagrangian.get_initial_path(args.calc_type,
                                        args.minima1,
                                        args.special_point,
                                        args.n_images,
                                        args.minima2,
                                        geodesic_code=args.geodesic_code,
                                        xyz_r_p=args.xyz_r_p,
                                        atomic_symbols=args.atomic_symbols,
                                        initial_path_type=args.initial_path_type)

        logger.info('Done with generating initial path generation.'
                    'Calling the path optimization.')

        if args.calc_type == 0:
            if args.opt_type == 1:
                from lag_opt_lib.steepest_descent import ase_opt
                args.minima1 = ase_opt(coords0=np.reshape(coords_r,
                                                          (len(atomic_symbols), 3)),
                                       labels=atomic_symbols,
                                       ml_model_path=args.ml_model_path)
                print('shape of min1', np.shape(args.minima1))
                args.minima2 = ase_opt(coords0=np.reshape(coords_p,
                                                          (len(atomic_symbols), 3)),
                                       labels=atomic_symbols,
                                       ml_model_path=args.ml_model_path)
                print('shape of min2', np.shape(args.minima2))
            elif args.opt_type == 0:
                args.minima1, _, _, _, _ =\
                    steepest_descent(coords_r,
                                     args.step_size_g_opt,
                                     args.eps_g_opt,
                                     args.max_iter_g_opt,
                                     calc_type=args.calc_type,
                                     labels=args.atomic_symbols,
                                     model_software=args.model_software,
                                     output_dir=str(args.output_dir),
                                     ml_model_path=str(args.ml_model_path))

                args.minima2, _, _, _, _ =\
                    steepest_descent(coords_p,
                                     args.step_size_g_opt,
                                     args.eps_g_opt,
                                     args.max_iter_g_opt,
                                     calc_type=args.calc_type,
                                     labels=args.atomic_symbols,
                                     model_software=args.model_software,
                                     output_dir=str(args.output_dir),
                                     ml_model_path=str(args.ml_model_path))

        '''
        from lag_opt_lib.compare import e_r_diff
        diff_r = e_r_diff([coords_r, coords_r_opt])
        print('Difference between reactant and optimized reactant:', diff_r)

        diff_r = e_r_diff([coords_p, coords_p_opt])
        print('Difference between product and optimized product:', diff_r)

        diff_r = e_r_diff([coords_r_opt, coords_p_opt])
        print('Difference between optimized reactant and optimized product:', diff_r)
        '''
        # import sys
        # sys.exit()

        result =\
            lagrangian.find_critical_path(
                args.calc_type,
                atomic_symbols=atomic_symbols,
                initial_points=coords_path,
                start=args.minima1,
                end=args.minima2,
                num_steps=args.max_iter_l_opt,
                step_factor=args.step_size_l_opt,
                action_type=args.action_type,
                path_grad_type=args.path_grad_type,
                a=args.a,
                convergence_type=args.convergence_type,
                eps=args.eps_l_opt,
                re_meshing_type=args.re_meshing_type,
                change_factor=args.change_factor,
                all_path_e_file=str(args.output_dir) + '/all_path_e.txt',
                output_path_file=str(args.output_dir) + '/output_path.txt',
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                ml_model_path=str(args.ml_model_path),
                re_mesh_frequency=args.re_mesh_frequency
            )

        print('Just before plots.')

        if args.calc_type == 0:
            plot_nth = args.nth
            plot_multiple_path_es(plot_nth,
                                  file_name=args.all_path_e_file,
                                  dist_file_name=str(args.output_dir) + '/path_dists.txt',
                                  output_dir=str(args.output_dir))
        elif args.calc_type == 1:
            if args.lag_opt_plot_type == 0:
                plot_contour(args.x_min,
                             args.x_max,
                             args.y_min,
                             args.y_max,
                             args.delta,
                             result,
                             calc_type=args.calc_type)
            elif args.lag_opt_plot_type == 1:
                # data = np.loadtxt(str(args.output_dir) + '/all_path_e.txt')
                data = np.loadtxt(str(args.output_dir) + '/lag_opt_path.txt')
                from lag_opt_lib.plot_functions import plot_gif
                plot_gif(data,
                         args.x_min,
                         args.x_max,
                         args.y_min,
                         args.y_max,
                         args.delta,
                         end1=args.minima1,
                         end2=args.minima2,
                         frame_frequency=args.frame_frequency,
                         calc_type=args.calc_type,
                         frames_per_second=args.frames_per_second,
                         output_dir=str(args.output_dir))
