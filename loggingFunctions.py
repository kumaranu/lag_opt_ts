import numpy as np
import os


def print_all_geoms(opt_path):
    if not os.path.exists('allGeoms'):
        os.mkdir('allGeoms')
    os.chdir('allGeoms')
    for i in range(len(opt_path)):
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        os.chdir(str(i))
        np.savetxt(str(i) + '.txt', opt_path[i], fmt='%.5f')
        os.chdir('../')
    os.chdir('../')


def save_all_geoms(atomic_symbols, opt_path, log_file='allGeoms.xyz'):
    file1 = open(log_file, 'a')
    n_atoms = len(opt_path[0])
    for i in range(len(opt_path)):
        file1.write(str(n_atoms) + '\n\n')
        for j in range(n_atoms):
            file1.write(atomic_symbols[j] + ' ' + str(opt_path[i][j][0]) + ' ' + str(opt_path[i][j][1]) +
                        ' ' + str(opt_path[i][j][2]) + '\n')
    file1.close()


def write_xyz(coords, symbols):
    n_atoms = len(symbols)
    coords = np.reshape(coords, (n_atoms, 3))
    file_str = str(n_atoms) + '\n'
    for i in range(n_atoms):
        file_str += '\n' + symbols[i] + ' ' + str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + str(coords[i][2])
    file_str += '\n'

    file1 = open('test.xyz', 'w')
    file1.write(file_str)
    file1.close()


def xyz_to_pdb():
    from openbabel import openbabel as ob
    # read the XYZ file
    xyz_file = "test.xyz"
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInFormat("xyz")
    mol = ob.OBMol()
    ob_conversion.ReadFile(mol, xyz_file)

    # write the PDB file
    pdb_file = "test.pdb"
    ob_conversion.SetOutFormat("pdb")
    ob_conversion.WriteFile(mol, pdb_file)
