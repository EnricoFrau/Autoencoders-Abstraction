import numpy as np
import torch
from AE.models import AE_0
from torchvision import datasets, transforms
from AE.datasets import MNISTDigit2OnlyDataset, MNISTDigit2Dataset, FEMNISTDataset


def compute_all_decoded_features_dist_over_repetitions(model_path_kwargs, model_kwargs, repetitions=range(6), datasets=None):

    distances_over_repetitions = np.array([])

    for i in repetitions:
        model_path_kwargs['dataset'] = datasets[0]
        model_path_kwargs['train_num'] = i
        model_D = load_model(model_path_kwargs, model_kwargs)

        for j in repetitions:
            model_path_kwargs['dataset'] = datasets[1]
            model_path_kwargs['train_num'] = j
            model_D_prime = load_model(model_path_kwargs, model_kwargs)

            distances = compute_all_decoded_features_distances(
                compute_all_decoded_features(model_D), 
                compute_all_decoded_features(model_D_prime)
            )

            distances_over_repetitions = np.append(distances_over_repetitions, distances)

    return distances_over_repetitions



def compute_all_decoded_features_distances_without_repetitions(model_path_kwargs, model_kwargs, repetitions = (0,0), datasets=None):
    model_path_kwargs['dataset'] = datasets[0]
    model_path_kwargs['train_num'] = repetitions[0]
    model_D = load_model(model_path_kwargs, model_kwargs)

    model_path_kwargs['dataset'] = datasets[1]
    model_path_kwargs['train_num'] = repetitions[1]
    model_D_prime = load_model(model_path_kwargs, model_kwargs)

    distances = compute_all_decoded_features_distances(
        compute_all_decoded_features(model_D), 
        compute_all_decoded_features(model_D_prime)
    )
    return distances

def compute_all_decoded_features_distances(all_features_D, all_features_D_prime):
    distances = np.zeros(len(all_features_D))
    for i in range(len(all_features_D)):
            distances[i] = calc_euclidean_distance(all_features_D[i], all_features_D_prime[i])
    return distances



def compute_all_decoded_features(model):
    all_features = []
    for i in range(model.latent_dim):
        feature_i = get_feature_i(i, model)
        all_features.append(feature_i)
    return all_features



def calc_euclidean_distance(feature_i_D, feature_i_D_prime):
    return np.linalg.norm(feature_i_D - feature_i_D_prime)




def get_feature_i(i, model):
    model.eval()
    with torch.no_grad():
        ld = model.latent_dim
        input_feature_state = torch.zeros((1, ld))
        input_feature_state[0, i] = 1
        feature_i = model.decode(input_feature_state).cpu().numpy()
        feature_i = feature_i.squeeze(0)  # Remove the first dimension
    return feature_i



def load_model(model_path_kwargs, model_kwargs):
    my_model = AE_0(
        **model_kwargs
    ).to(model_kwargs['device'])
    model_path = f"../models/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['train_type']}/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/dr{model_path_kwargs['decrease_rate']}_{model_path_kwargs['num_hidden_layers']}hl_{model_path_kwargs['train_num']}.pth"
    my_model.load_state_dict(torch.load(model_path, map_location=model_kwargs['device']))
    return my_model





# ----------------------------------------------------------------------------








def compute_min_distances_frequencies(model_path_kwargs, model_kwargs, datapoints_array, labels_array):
    decoded_binary_matrix = compute_decoded_binary_matrix(model_path_kwargs, model_kwargs)
    closest_rows_indices = compute_closest_rows(decoded_binary_matrix, datapoints_array)
    datapoints_frequencies = compute_min_dist_datapoints_frequencies(closest_rows_indices, dataset_length=datapoints_array.shape[0])
    labels_frequencies = compute_min_dist_labels_frequencies(closest_rows_indices, labels_array)
    return datapoints_frequencies, labels_frequencies



# ----------------------------------------------------------------------------



def compute_decoded_binary_matrix(model_path_kwargs, model_kwargs):

    model = load_model(model_path_kwargs, model_kwargs)
    model.eval()
    with torch.no_grad():
        latent_matrix = torch.tensor(binary_matrix(model.latent_dim))
        decoded_matrix = model.decode(latent_matrix).cpu().numpy()
        return decoded_matrix
    


def compute_closest_rows(decoded_binary_matrix, datapoints_array):
    return np.array([find_closest_row(vec, datapoints_array) for vec in decoded_binary_matrix])



def compute_min_dist_datapoints_frequencies(closest_rows_indices, dataset_length=60000):
    datapoints_frequencies_array = np.zeros(dataset_length, dtype=np.float64)
    unique, counts = np.unique(closest_rows_indices, return_counts=True)
    datapoints_frequencies_array[unique] = counts
    # datapoints_frequencies_array /= np.sum(datapoints_frequencies_array)
    return datapoints_frequencies_array



def compute_min_dist_labels_frequencies(closest_rows_indices, labels_array):
    unique, counts = np.unique(labels_array[closest_rows_indices], return_counts=True)
    labels_frequencies_array = np.zeros(np.max(unique)+1, dtype=np.float64)
    labels_frequencies_array[unique] = counts
    # labels_frequencies_array /= np.sum(labels_frequencies_array)
    return labels_frequencies_array



# ----------------------------------------------------------------------------



def binary_matrix(n):
    """
    Returns a numpy array of shape (2**n, n) where each row is the binary representation of numbers from 0 to 2**n-1.
    """
    numbers = np.arange(2**n)
    # Convert each number to binary and pad with zeros to length n
    bin_matrix = ((numbers[:, None] & (1 << np.arange(n)[::-1])) > 0).astype(np.float32)
    return bin_matrix



def find_closest_row(vector, matrix):
    """
    Returns the row index of the matrix whose row has the minimum Euclidean distance to the input vector.

    Args:
        vector (np.ndarray): 1D array of length n.
        matrix (np.ndarray): 2D array of shape (m, n).

    Returns:
        int: Index of the closest row.
    """
    # Compute Euclidean distances for each row
    distances = np.linalg.norm(matrix - vector, axis=1)
    # Return the index of the minimum distance
    return np.argmin(distances)



#----------------------------------------------------------------------------





def get_datapoints_labels_arrays(dataset_name):
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(
            '/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == 'EMNIST':
        dataset = datasets.EMNIST(
            '/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data',
            split='balanced',
            train=True,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == '2MNISTonly':
        dataset = MNISTDigit2OnlyDataset(train=True, download=True)
    elif dataset_name == '2MNIST':
        dataset = MNISTDigit2Dataset(train=True, download=True, target_size=60000)
    elif dataset_name == 'FEMNIST':
        dataset = FEMNISTDataset(train=True, download=True)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    datapoints_array = []
    labels_array = []
    for img, lbl in dataset:
        datapoints_array.append(img.numpy().flatten())
        labels_array.append(lbl)
    datapoints_array = np.array(datapoints_array)
    labels_array = np.array(labels_array)

    sorting_permutation = np.argsort(labels_array)

    labels_array = labels_array[sorting_permutation]
    datapoints_array = datapoints_array[sorting_permutation]

    return datapoints_array, labels_array
