import numpy as np
import nibabel as nib

from sklearn.decomposition import IncrementalPCA

from os import listdir
from os.path import join


from pickle import dump, load
import pdb

# experiment_name = "Stacked_16_16_16_16"
channel_size = 16

def load_mask():
    # mask_path = "/data/mask/average_optthr.nii"
    mask_path = "/data_berlin/mask/binary_mask4grey_BerlinMargulies26subjects.nii"
    mask = nib.load(mask_path).get_data().ravel()
    mask_idx = mask.nonzero()[0]
    return mask_idx


def ipca(n_components):
    # train_directory = "/data/train"
    train_directory = "/data_cobre/train"
    # train_directory = "/out/{}/feature".format(experiment_name)
    train_file = sorted(listdir(train_directory))
    mask_idx = load_mask()
    ipca = IncrementalPCA(n_components=n_components)
    batch_num_subject = 67
    num_of_batch = len(train_file) // batch_num_subject + 1
    batch_size = 150 * batch_num_subject
    if batch_size < n_components:
        raise RuntimeError("batch_size must be larger than n_components")
    for batch_idx in range(num_of_batch):
        subject_idx_min = batch_idx * batch_num_subject
        subject_idx_max = min((batch_idx + 1) * batch_num_subject, len(train_file))
        print("processing {} ... {} / {}".format(subject_idx_min, subject_idx_max, len(train_file)))
        batch_subject = range(subject_idx_min, subject_idx_max)
        stacked_ndarray = np.empty((150 * len(batch_subject), len(mask_idx)), dtype=np.float32)
        # stacked_ndarray = np.empty((150 * len(batch_subject), channel_size * 9 * 11 * 10), dtype=np.float32)
        for i, subject in enumerate(batch_subject):
            print("loading {}".format(subject))
            data = nib.load(join(train_directory, train_file[subject])).get_data()
            # data = np.load(join(train_directory, train_file[subject])).reshape((150, -1))
            assert isinstance(data, np.ndarray)
            # data_masked = data
            data_masked = data.reshape((91 * 109 * 91, 150))[mask_idx, :].T
            stacked_ndarray[150 * i : 150 * (i + 1), :] = data_masked
        print("fitting started")
        ipca.partial_fit(stacked_ndarray)
        print("fitting ended")
    return ipca


def reduction(ipca_orig, n_components):
    noise = 0

    mean_ = np.copy(ipca_orig.mean_)
    whiten = ipca_orig.whiten
    components_ = np.copy(ipca_orig.components_)[:n_components, :]
    explained_variance_ = np.copy(ipca_orig.explained_variance_)[:n_components]

    test_directory = "/data_cobre/test"
    # test_directory = "/out/{}/feature".format(experiment_name)
    test_file = sorted(listdir(test_directory))
    mask_idx = load_mask()
    stack = []
    for subject in range(len(test_file)):
        print("{}/{}".format(subject, len(test_file)))
        data = nib.load(join(test_directory, test_file[subject])).get_data()
        # data = np.load(join(test_directory, test_file[subject])).reshape((150, -1))
        data_masked_orig = data.reshape((91 * 109 * 91, -1))[mask_idx, :].T
        # data_masked = data
        data_masked = np.copy(data_masked_orig)
        if mean_ is not None:
            data_masked = data_masked - mean_
        data_masked += noise * np.random.randn(*data_masked.shape)

        data_transformed = np.dot(data_masked, components_.T)
        if whiten:
            data_transformed /= np.sqrt(explained_variance_)
        # pdb.set_trace()
        if whiten:
            data_reconstructed = np.dot(data_transformed, np.sqrt(explained_variance_[:, np.newaxis]) *
                            components_) + mean_
        else:
            data_reconstructed = np.dot(data_transformed, components_) + mean_

        absolute_error = np.mean(np.abs(data_masked - data_reconstructed))
        print(absolute_error)
        stack.append(absolute_error)
    return stack

# np.mean(stack) == 0.069458835
def extract():
    n_components = 10000
    i = ipca(n_components=n_components)
    # with open("/efs/fMRI_AE/ipca_{}.pickle".format(n_components), "wb") as f:
    try:
        with open("/out/ipca_{}.pickle".format(n_components), "wb") as f:
            dump(i, f, protocol=4)
    except:
        pdb.set_trace()


def deduce():
    n_components_name = 10000
    # with open("/efs/fMRI_AE/ipca_{}.pickle".format(n_components), "rb") as f:
    with open("/efs/fMRI_AE/ipca_{}.pickle".format(n_components_name), "rb") as f:
        i = load(f)
    n_components = 10000
    s = reduction(i, n_components)
    # print(s)
    print(np.mean(s))


if __name__ == "__main__":
    deduce()