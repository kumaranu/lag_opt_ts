##########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, utils, torch_tools

model = '/home/kumaranu/PycharmProjects/optimization_0/MACE_model_cpu.model'
configs = '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/init_path.xyz'
output = '/home/kumaranu/PycharmProjects/optimization_0/output_test.txt'

from ase import Atoms
from readIn import read_in_multiple_xyz_file

atomic_symbols, xyz_coords_list = read_in_multiple_xyz_file(configs, n_images=40)

atomic_numbers = [ase.data.atomic_numbers[s] for s in atomic_symbols]

list_of_ase_configs = []
for i in range(len(xyz_coords_list)):
    list_of_ase_configs.append(Atoms(numbers=atomic_numbers, positions=xyz_coords_list[i]))

def getE(model, list_of_ase_configs, default_dtype='double', device='cpu', batch_size=64,
         compute_stress=False):
    '''

    :param model: type(str), Required argument, path to the model
    :param configs: type(str), Required argument, path to the xyz configurations
    :param output: type(str), Required argument, path to the output of the model
    :param default_dtype: type(str), choices[float32, float64, double (default)]
    :param device: type(str), choices['cpu (default), 'cuda']
    :param batch_size: type(int), batch size, default=64
    :param compute_stress: type(bool), compute stress, default=False
    :param return_contributions: type(bool), model outputs energy contributions for each body order,
    only supported for MACE, not ScaleShiftMACE, default=False
    :param info_prefix: type(str), prefix for energy, forces and stress keys, default=MACE_
    :return: None
    '''

    torch_tools.set_default_dtype(default_dtype)
    device = torch_tools.init_device(device)

    # Load model
    model = torch.load(model, map_location=device)

    configs = [data.config_from_atoms(atoms) for atoms in list_of_ase_configs]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max)) for config in configs],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        forces = np.split(torch_tools.to_numpy(output["forces"]), indices_or_sections=batch.ptr[1:], axis=0)
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [forces for forces_list in forces_collection for forces in forces_list]
    return [energies, forces_list]


if __name__ == '__main__':
    energy_list, forces_list = getE(model, list_of_ase_configs, default_dtype='float32')
