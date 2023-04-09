from ase.io import read
import geodesic_interpolate
import numpy as np

def e_r_diff(atoms):
    wij_list = []
    for atom in atoms:
        labels = atom.get_chemical_symbols()
        geom   = atom.get_positions()
        rijlist, re = geodesic_interpolate.coord_utils.get_bond_list(geom)
        scaler = geodesic_interpolate.coord_utils.morse_scaler(alpha=0.7, re=re)
        wij, _ = geodesic_interpolate.coord_utils.compute_wij(geom, rijlist, scaler)
        wij_list.append(wij)
    diff_r = np.linalg.norm(wij_list[0] - wij_list[1])
    return diff_r

filename = '/home/kumaranu/Documents/testing_geodesic/inputs/abcd/A_B.xyz'
atoms = read(filename, index=':')

diff_r = e_r_diff(atoms)
print(diff_r)



