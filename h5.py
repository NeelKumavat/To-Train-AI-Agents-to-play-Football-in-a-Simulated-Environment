import h5py
import numpy as np
import os

def save_genomes(genomes, filename='genomes.h5'):
    """
    Save the list of genome arrays into an HDF5 file.
    """
    # Stack genomes into a 2D array (each row is one genome)
    genomes_array = np.stack(genomes)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('genomes', data=genomes_array)

def load_genomes(filename='genomes.h5'):
    """
    Load genomes from an HDF5 file.
    Returns a list of genome arrays or None if file not found.
    """
    if not os.path.exists(filename):
        return None
    with h5py.File(filename, 'r') as hf:
        genomes_array = hf['genomes'][:]
    # Convert each row back into a separate numpy array
    return [genomes_array[i] for i in range(genomes_array.shape[0])]
