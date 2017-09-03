import os
import sys
import time
import pickle
import random
import tarfile
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage
from urllib.request import urlretrieve
from sklearn.linear_model import LogisticRegression


eps = 1e-5

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

num_classes = 10
np.random.seed(133)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_size = 200000
valid_size = 10000
test_size = 10000


def show_image_float(img, cmap=None):
    #plt.imshow(img, norm=matplotlib.colors.NoNorm(vmin=-0.5, vmax=0.5), cmap='gray')
    plt.imshow(img, vmin=-0.5, vmax=0.5, cmap='gray')
    plt.show()

def show_image_filesystem(filename):
    img = mpimg.imread(filename)
    plt.imshow(img, cmap='gray')
    plt.show()

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('%s%%' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)

                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def verify_dataset(dataset, labels):
    idx = random.randrange(0, labels.shape[0])
    print(idx, labels[idx], chr(ord('a') + labels[idx]))
    plt.imshow(dataset[idx], cmap='gray')
    plt.show()

def generate_datasets(pickle_file):
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    print(train_filename, test_filename)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    print(train_folders, test_folders)

    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

    print(train_datasets, test_datasets)

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size
    )

    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    if not os.path.isfile(pickle_file):
        try:
            with open(pickle_file, 'wb') as fobj:
                save = {
                    'train_dataset': train_dataset,
                    'train_labels': train_labels,
                    'valid_dataset': valid_dataset,
                    'valid_labels': valid_labels,
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                }
                pickle.dump(save, fobj, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

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

def find_duplicates(dataset1, labels1, dataset2, labels2, threshold=eps):
    means = np.mean(dataset1, axis=(1,2))
    indices = means.argsort()
    num_groups = 2000
    indices = np.reshape(indices, (num_groups, indices.shape[0] // num_groups))

    duplicate_indices = []
    dataset2_idx = 0
    for item in dataset2:
        belongs_to_group = 0
        item_mean = item.mean()
        for group_idx in range(num_groups):
            if item_mean >= means[indices[group_idx][0]]:
                belongs_to_group = group_idx
            else:
                break
        for idx_in_group in range(len(indices[belongs_to_group])):
            dataset_idx = indices[belongs_to_group][idx_in_group]
            mse = ((item - dataset1[dataset_idx]) ** 2).mean()
            if mse < threshold:
                if labels1[dataset_idx] == labels2[dataset2_idx]:
                    pass
                else:
                    print(
                        'Found duplicate! Labels are different',
                        labels1[dataset_idx],
                        labels2[dataset2_idx],
                    )
                duplicate_indices.append(dataset2_idx)
                # show_image_float(item)
                # show_image_float(dataset1[dataset_idx])
                break

        dataset2_idx += 1
        if dataset2_idx % 100 == 0:
            print('Processed', dataset2_idx, 'duplicates:', len(duplicate_indices))

    print('Found total of', len(duplicate_indices), 'duplicantes!')
    return duplicate_indices

def sanitize_datasets(datasets):
    train_dataset, train_labels = extract_dataset(datasets, 'train')
    test_dataset, test_labels = extract_dataset(datasets, 'valid')
    valid_dataset, valid_labels = extract_dataset(datasets, 'test')

    duplicate_indices = find_duplicates(
        train_dataset, train_labels, test_dataset, test_labels, threshold=0.005,
    )
    test_dataset_sanitized = np.delete(test_dataset, duplicate_indices, axis=0)
    test_labels_sanitized = np.delete(test_labels, duplicate_indices, axis=0)

    duplicate_indices = find_duplicates(
        train_dataset, train_labels, valid_dataset, valid_labels, threshold=0.005,
    )
    valid_dataset_sanitized = np.delete(valid_dataset, duplicate_indices, axis=0)
    valid_labels_sanitized = np.delete(valid_labels, duplicate_indices, axis=0)

    pickle_file_sanitized = os.path.join(data_root, 'notMNIST_sanitized.pickle')
    try:
        with open(pickle_file_sanitized, 'wb') as fobj:
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset_sanitized,
                'valid_labels': valid_labels_sanitized,
                'test_dataset': test_dataset_sanitized,
                'test_labels': test_labels_sanitized,
            }
            pickle.dump(save, fobj, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file_sanitized, ':', e)
        raise

def train_logistic_classifier(x, y, x_test, y_test, num_train_samples=None):
    tstart = time.time()
    if num_train_samples is None:
        num_train_samples = x.shape[0]
    x = x[:num_train_samples]
    y = y[:num_train_samples]

    resolution = x.shape[1]
    x = x.reshape(num_train_samples, resolution * resolution)
    x_test = x_test.reshape(x_test.shape[0], resolution * resolution)

    classifier = LogisticRegression(
        C=100./num_train_samples,
        multi_class='multinomial',
        penalty='l2',
        solver='saga',
        max_iter=200,
    )
    classifier.fit(x, y)
    sparsity = np.mean(classifier.coef_ == 0) * 100
    score_train = classifier.score(x, y)
    score_test = classifier.score(x_test, y_test)
    print('Sparsity: %.2f%%' % sparsity)
    print('Train score: %.4f' % score_train)
    print('Test score: %.4f' % score_test)

    interactive = True
    if interactive:
        for i in range(10):
            image = x_test[i]
            prediction = classifier.predict(image.reshape((1, resolution * resolution)))[0]
            prediction_proba = classifier.predict_proba(image.reshape((1, resolution * resolution)))
            print('Prediction:', prediction, chr(ord('a') + prediction))
            print('Prediction proba:', prediction_proba)
            show_image_float(image.reshape((resolution, resolution)))

    coef = classifier.coef_.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(10):
        plot = plt.subplot(2, 5, i + 1)
        plot.imshow(
            coef[i].reshape(resolution, resolution),
            interpolation='nearest',
            cmap=plt.cm.RdBu,
            vmin=-scale,
            vmax=scale,
        )
        plot.set_xticks(())
        plot.set_yticks(())
        plot.set_xlabel('Class %i' % i)
    plt.suptitle('Classification vector for...')

    run_time = time.time() - tstart
    print('Example run in %.3f s' % run_time)
    plt.show()

def main():
    pickle_file = os.path.join(data_root, 'notMNIST_sanitized.pickle')
    if not os.path.isfile(pickle_file):
        generate_datasets(pickle_file)

    datasets = load_datasets(pickle_file)

    train_dataset, train_labels = extract_dataset(datasets, 'train')
    test_dataset, test_labels = extract_dataset(datasets, 'valid')
    valid_dataset, valid_labels = extract_dataset(datasets, 'test')

    print(train_dataset.shape, train_labels.shape)
    print(test_dataset.shape, test_labels.shape)
    print(valid_dataset.shape, valid_labels.shape)

    def train(samples=None):
        train_logistic_classifier(
            train_dataset,
            train_labels,
            test_dataset,
            test_labels,
            num_train_samples=samples,
        )

    train(300)

    return 0


if __name__ == '__main__':
    sys.exit(main())