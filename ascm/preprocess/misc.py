import os
import pickle

def pickle_dump(data, folder_dir, file_name='data.pkl'):
    """Dump data as a pickle file."""
    os.makedirs(folder_dir, exist_ok=True)
    dir = os.path.join(folder_dir, file_name)
    with open(dir, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(folder_dir, file_name='data.pkl'):
    """Load data from a pickle file."""
    dir = os.path.join(folder_dir, file_name)
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    return data