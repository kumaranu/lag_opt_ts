import numpy as np


def line_plot(segment_i, segment_f, check_segment):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(segment_i, segment_f), check_segment)
    plt.title('Line Plot Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def get_n_images():
    from glob import glob
    list_xyz_files = glob('lag_opt_path*.xyz')
    
    all_file_names = []
    for file in list_xyz_files:
        all_file_names.append(int(file[12:].split('.')[0]))
    
    return np.max(all_file_names) + 1


def get_check_segment(segment_i, segment_f, all_path_energy):
    check_segment = []
    for point in range(segment_i, segment_f):
        check_segment.append(all_path_energy[-1][point])
    return path_segment_i + np.argmax(check_segment), check_segment


def read_in_xyz_file(file_name, config_num):
    from ase.io import read, write
    atoms = read(file_name, index=':')
    config = atoms[config_num]
    write(f'config.xyz', config)


def sella_wrapper(xyz_file_path, ml_model_path):
    from ase.io import read
    from mace.calculators import MACECalculator
    from sella import Sella

    calculator = MACECalculator(model_path=ml_model_path, device='cpu')
    atoms = read(xyz_file_path, '0')
    atoms.set_calculator(calculator)

    opt = Sella(atoms, internal=True, trajectory='A_C+D.traj')
    opt.run(fmax=0.000001)

    input_file, output_file = 'A_C+D.traj', 'A_C+D.xyz'
    convert_traj_to_xyz(input_file, output_file)


def convert_traj_to_xyz(input_file, output_file):
    from ase.io import read, write
    atoms = read(input_file, index=":", format="traj")
    write(output_file, atoms, format="xyz")


file_path = 'config.xyz'
model_path = '/home/kumaranu/PycharmProjects/optimization_0/MACE_model_cpu_double.model'
all_path_e_file = 'all_path_e'
path_segment_i, path_segment_f = 14, 18
n_images = get_n_images()
filename = 'lag_opt_path' + str(n_images - 1) + '.xyz'

x = np.arange(path_segment_i, path_segment_f)
all_path_e = np.loadtxt(all_path_e_file)

approx_ts_loc, y = get_check_segment(path_segment_i, path_segment_f, all_path_e)
read_in_xyz_file(filename, approx_ts_loc)
# line_plot(x,y)

sella_wrapper(file_path, model_path)
