from compare_geoms.molecule_processing import process_molecules
from compare_geoms.data_loading import load_data_from_h5, load_reference_molecule

sample_h5_file = 'output_9953.h5'
sample_ref_molecule = '264_noise00.xyz'


def test_process_molecules():
    # Test the process_molecules function
    # Create dummy data for testing
    ref_coords, ref_atomic_nums, ref_molecule = load_reference_molecule(sample_ref_molecule)
    data_list = load_data_from_h5(sample_h5_file, ref_coords, ref_atomic_nums, ref_molecule)
    result = process_molecules(data_list)
