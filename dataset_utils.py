import pickle
import numpy as np


PICKLE_FILE = 'notMNIST.pickle'
PICKLE_FILE_SANITIZED = 'notMNIST_sanitized.pickle'

NUM_CLASSES = 10
IMAGE_RES = 28  # pixel width and height


def load_datasets(pickle_file):
    try:
        with open(pickle_file, 'rb') as fobj:
            datasets = pickle.load(fobj)
    except Exception as e:
        print('Unable to process data from', pickle_file, ':', e)
        raise
    return datasets

def extract_dataset(datasets, name):
    return datasets[name + '_dataset'], datasets[name + '_labels']

def flatten_dataset(data, labels):
    assert data.ndim == 3
    assert data.shape[1] == data.shape[2]
    data = data.reshape((-1, data.shape[1] * data.shape[2])).astype(np.float32)
    labels = (np.arange(NUM_CLASSES) == labels[:, None]).astype(np.float32)
    return data, labels

def get_flattened_dataset(datasets, name):
    data, labels = extract_dataset(datasets, name)
    return flatten_dataset(data, labels)
