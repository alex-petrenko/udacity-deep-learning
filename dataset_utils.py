import pickle
import numpy as np

from tensorflow.contrib.data import Dataset


PICKLE_FILE = 'notMNIST.pickle'
PICKLE_FILE_SANITIZED = 'notMNIST_sanitized.pickle'

NUM_CLASSES = 10
IMAGE_RES = 28  # pixel width and height


def load_datasets(pickle_file):
    try:
        with open(pickle_file, 'rb') as fobj:
            datasets = pickle.load(fobj)
    except Exception as exc:
        print('Unable to process data from', pickle_file, ':', exc)
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

def image_dataset(data, labels):
    assert data.ndim == 3
    assert data.shape[1] == data.shape[2]
    data = data.reshape((-1, data.shape[1], data.shape[2], 1)).astype(np.float32)
    labels = (np.arange(NUM_CLASSES) == labels[:, None]).astype(np.float32)
    return data, labels

def get_image_dataset(datasets, name):
    data, labels = extract_dataset(datasets, name)
    return image_dataset(data, labels)

def dataset_to_inputs(data, labels, batch_size):
    """Returns tuple (input_tf_node, labels_tf_node, iterator)."""
    dataset = Dataset.from_tensor_slices({'x': data, 'y': labels})
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    sample = iterator.get_next()
    x = sample['x']
    y = sample['y']
    return x, y, iterator
