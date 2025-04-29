import numpy as np
from ase.io import read, write
from newtonnet.utils.ase_interface import MLAseCalculator as NewtonNet
import copy


def newton_raphson(
        atoms,
        newtonnet_model_path,
        newtonnet_config_path,
        max_iter=100,
        threshold=0.0000001,
        ):
    count = 0
    natoms = len(atoms)
    trajectory = [copy.deepcopy(atoms)]
    while count < max_iter:
        mlcalculator = NewtonNet(
            model_path=newtonnet_model_path.split(":"),
            settings_path=newtonnet_config_path.split(":"),
        )
        mlcalculator.calculate(atoms)
        hessian = np.reshape(mlcalculator.results['hessian'], (3 * natoms, 3 * natoms))
        eigenvalues, eigenvectors = np.linalg.eig(hessian)

        print('count, energy, imag_eigval:',
              count, mlcalculator.results['energy'], np.min(eigenvalues.imag))

        gradient = -np.reshape(mlcalculator.results['forces'], (3 * natoms,))

        dr = np.zeros((3 * natoms, 1))
        for i in range(3 * natoms - 6):
            dr += np.reshape(eigenvectors.real[:, i] * (np.dot(np.transpose(eigenvectors.real[:, i]),
                                                    gradient) / eigenvalues.real[i]), (3 * natoms, 1))
        r = copy.deepcopy(atoms.arrays['positions'])
        atoms.arrays['positions'] = r - np.reshape(dr, (natoms, 3))
        trajectory.append(copy.deepcopy(atoms))
        if np.linalg.norm(dr) < threshold:
            print(f'TS optimization converged in {count} steps.')
            break
        count += 1
    return atoms, trajectory


np.set_printoptions(linewidth=np.inf, precision=1, threshold=np.inf, suppress=True)


if __name__ == '__main__':
    newtonnet_model_path = '/home/kumaranu/Documents/NewtonNet/example/predict/training_52/models/best_model_state.tar:' \
                           '/home/kumaranu/Documents/NewtonNet/example/predict/training_53/models/best_model_state.tar:' \
                           '/home/kumaranu/Documents/NewtonNet/example/predict/training_54/models/best_model_state.tar:' \
                           '/home/kumaranu/Documents/NewtonNet/example/predict/training_55/models/best_model_state.tar'
    newtonnet_config_path = '/home/kumaranu/Documents/NewtonNet/example/predict/training_52/run_scripts/config0.yml:' \
                            '/home/kumaranu/Documents/NewtonNet/example/predict/training_52/run_scripts/config0.yml:' \
                            '/home/kumaranu/Documents/NewtonNet/example/predict/training_52/run_scripts/config0.yml:' \
                            '/home/kumaranu/Documents/NewtonNet/example/predict/training_52/run_scripts/config0.yml'
    atoms = read('/home/kumaranu/Downloads/sella_ts/singlets/002.xyz')
    ts_atoms, ts_trajectory = newton_raphson(
        atoms,
        newtonnet_model_path,
        newtonnet_config_path
    )

    write('/home/kumaranu/Downloads/ts_trajectory.xyz', ts_trajectory)
    mlcalculator = NewtonNet(
        model_path=newtonnet_model_path.split(":"),
        settings_path=newtonnet_config_path.split(":"),
    )
    mlcalculator.calculate(ts_atoms)
    natoms = len(ts_atoms)
    hessian = np.reshape(mlcalculator.results['hessian'], (3 * natoms, 3 * natoms))
    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    print('eigenvalues:', eigenvalues)
