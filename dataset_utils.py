import pickle


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
