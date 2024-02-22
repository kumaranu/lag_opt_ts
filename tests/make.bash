#!/bin/bash

python main.py \
--input_dir '/home/kumaranu/PycharmProjects/optimization_0/inputs' \
--output_dir '/home/kumaranu/PycharmProjects/optimization_0/outputs' \
--job_type ts_lag \
--opt_type 1 \
--n_images 40 \
--calc_type 0 \
--max_iter_g_opt 100 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 200 \
--step_size_l_opt 0.001 \
--eps_l_opt 1e-9 \
--nth 20 \
--a 150 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--re_meshing_type 1 \
--re_mesh_frequency 15 \
--frame_frequency 100 \
--frames_per_second 1 \
--convergence_type 0 \
--ml_model_path '/home/kumaranu/PycharmProjects/optimization_0/inputs/ml_models/MACE_model_cpu_double.model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd1/34/34.xyz'

exit 1
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd1/12/12.xyz'
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd1/13/13.xyz'
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd1/23/23.xyz'
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd1/34/34.xyz'
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/B_C+D.xyz'
--xyz_r_p '/home/kumaranu/PycharmProjects/optimization_0/inputs/A_B_optimized.xyz'

python main.py \
--calc_type 1 \
--max_iter_g_opt 2000 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 100 \
--step_size_l_opt 0.0002 \
--eps_l_opt 1e-9 \
--job_type ts_lag \
--n_images 10 \
--nth 200 \
--a 150 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_C+D.xyz'


python main.py \
--calc_type 0 \
--max_iter_g_opt 2000 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 1000 \
--step_size_l_opt 0.0001 \
--eps_l_opt 1e-9 \
--job_type ts_lag \
--n_images 40 \
--nth 200 \
--a 200 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_C+D.xyz'

exit 1

python main.py \
--calc_type 0 \
--max_iter_g_opt 2000 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 1000 \
--step_size_l_opt 0.0001 \
--eps_l_opt 1e-9 \
--job_type ts_lag \
--n_images 40 \
--nth 200 \
--a 200 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/B_C+D.xyz'

exit 1

python main.py \
--calc_type 0 \
--max_iter_g_opt 2000 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 1000 \
--step_size_l_opt 0.0001 \
--eps_l_opt 1e-9 \
--job_type ts_lag \
--n_images 40 \
--nth 200 \
--a 200 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_B.xyz'

python main.py \
--calc_type 0 \
--max_iter_g_opt 2000 \
--step_size_g_opt 0.002 \
--eps_g_opt 1e-7 \
--max_iter_l_opt 200 \
--step_size_l_opt 0.0001 \
--eps_l_opt 1e-7 \
--job_type ts_lag \
--n_images 20 \
--nth 10 \
--a 200 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model' \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_B.xyz'

python main.py \
--calc_type 0 \
--max_iter 2000 \
--step_size_g_opt 0.02 \
--eps_l_opt 1e-7 \
--step_size_l_opt 0.02 \
--eps_l_opt 1e-7 \
--job_type opt \
--n_images 20 \
--nth 10 \
--a 500 \
--action_type 1 \
--path_grad_type 1 \
--model_software 'mace_model'


python main.py \
--calc_type 0 \
--max_iter 800 \
--step_size 0.001 \
--job_type ts_lag \
--n_images 20 \
--nth 40 \
--a 200 \
--action_type 1 \
--path_grad_type 1 \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_B.xyz'


python main.py \
--calc_type 0 \
--max_iter 100 \
--eps 1e-9 \
--step_size 0.0001 \
--job_type ts_lag \
--n_images 80 \
--nth 1 \
--a 5 \
--action_type 1 \
--path_grad_type 1 \
--xyz_r_p '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/B_C+D.xyz'




