import numpy as np
import pickle
import torch
import torch.nn as nn
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




def compute_min_distances_frequencies(model_path_kwargs, model_kwargs, datapoints_array, labels_array, return_distances=False):
    """
    Computes the frequency with which each datapoint and label is the closest to a decoded binary latent vector.

    Loads the model specified by model_path_kwargs and model_kwargs, decodes all possible binary latent vectors,
    finds the closest datapoint in datapoints_array for each decoded vector, and counts how often each datapoint
    and label is selected as the closest. Optionally returns the minimum distances.

    Args:
        model_path_kwargs (dict): Dictionary of model path parameters.
        model_kwargs (dict): Dictionary of model initialization parameters.
        datapoints_array (np.ndarray): Array of shape (n_datapoints, input_dim), containing datapoints.
        labels_array (np.ndarray): Array of shape (n_datapoints,), containing labels for each datapoint.
        return_distances (bool, optional): If True, also returns the array of minimum distances.

    Returns:
        tuple:
            - np.ndarray: Array of shape (n_datapoints,) with frequencies for each datapoint.
            - np.ndarray: Array of shape (n_labels,) with frequencies for each label.
            - np.ndarray (optional): Array of shape (2**latent_dim,) with minimum distances for each decoded vector.
    """
    decoded_binary_matrix = compute_decoded_binary_matrix(model_path_kwargs, model_kwargs)
    closest_rows_indices = compute_closest_rows(decoded_binary_matrix, datapoints_array)

    if return_distances:
        distances = np.linalg.norm(decoded_binary_matrix - datapoints_array[closest_rows_indices], axis=1)

    datapoints_frequencies = compute_min_dist_datapoints_frequencies(closest_rows_indices, dataset_length=datapoints_array.shape[0])
    labels_frequencies = compute_min_dist_labels_frequencies(closest_rows_indices, labels_array)

    if return_distances:
        return datapoints_frequencies, labels_frequencies, distances
    else:
        return datapoints_frequencies, labels_frequencies


def compute_min_distances(model_path_kwargs, model_kwargs, datapoints_array):
    """
    Computes the minimum Euclidean distances between each decoded binary latent vector and its closest datapoint.

    Loads the model specified by model_path_kwargs and model_kwargs, decodes all possible binary latent vectors,
    finds the closest datapoint in datapoints_array for each decoded vector, and returns the array of minimum distances.

    Args:
        model_path_kwargs (dict): Dictionary of model path parameters.
        model_kwargs (dict): Dictionary of model initialization parameters.
        datapoints_array (np.ndarray): Array of shape (n_datapoints, input_dim), containing datapoints.

    Returns:
        np.ndarray: Array of shape (2**latent_dim,) containing the minimum distances for each decoded vector.
    """
    decoded_binary_matrix = compute_decoded_binary_matrix(model_path_kwargs, model_kwargs)
    closest_rows_indices = compute_closest_rows(decoded_binary_matrix, datapoints_array)
    distances = np.linalg.norm(decoded_binary_matrix - datapoints_array[closest_rows_indices], axis=1)
    return distances


# ----------------------------------------------------------------------------



def compute_decoded_binary_matrix(model_path_kwargs, model_kwargs):
    """
    Decodes all possible binary latent vectors for a given model.

    Loads the model specified by model_path_kwargs and model_kwargs, creates a binary matrix of all possible
    latent states (shape: [2**latent_dim, latent_dim]), decodes each latent vector using the model, and returns
    the resulting decoded matrix.

    Args:
        model_path_kwargs (dict): Dictionary of model path parameters.
        model_kwargs (dict): Dictionary of model initialization parameters.

    Returns:
        np.ndarray: Decoded matrix of shape (2**latent_dim, input_dim), where each row is the decoded output
                    for a binary latent vector.
    """
    model = load_model(model_path_kwargs, model_kwargs)
    model.eval()
    with torch.no_grad():
        latent_matrix = torch.tensor(binary_matrix(model.latent_dim))
        decoded_matrix = model.decode(latent_matrix).cpu().numpy()
        return decoded_matrix
    


def compute_closest_rows(decoded_binary_matrix, datapoints_array):
    """
    For each row in the decoded_binary_matrix, finds the index of the closest row in datapoints_array
    using minimum Euclidean distance.

    Args:
        decoded_binary_matrix (np.ndarray): Array of shape (n, d), where each row is a decoded feature vector.
        datapoints_array (np.ndarray): Array of shape (m, d), where each row is a datapoint.

    Returns:
        np.ndarray: Array of shape (n,) containing the indices of the closest datapoint in datapoints_array
                    for each row in decoded_binary_matrix.
    """
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





def get_datapoints_labels_arrays(dataset_name, train=True):
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(
            '/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == 'EMNIST':
        dataset = datasets.EMNIST(
            '/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data',
            split='balanced',
            train=train,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == '2MNISTonly':
        dataset = MNISTDigit2OnlyDataset(train=train, download=True)
    elif dataset_name == '2MNIST':
        dataset = MNISTDigit2Dataset(train=train, download=True, target_size=60000)
    elif dataset_name == 'FEMNIST':
        dataset = FEMNISTDataset(train=train, download=True)
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




def plot_labels_frequencies_histogram(labels_frequencies, first_indices=None, title=None, cmap_name='inferno', ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    n = len(labels_frequencies)
    cmap = plt.get_cmap(cmap_name)

    if first_indices is not None:
        indices = [int(x) for x in first_indices]
        indices.append(n)
        num_segments = len(indices) - 1
        colors = [cmap(i / max(num_segments - 1, 1)) for i in range(num_segments)]
        for i in range(num_segments):
            start = indices[i]
            end = indices[i+1]
            ax.bar(range(start, end), labels_frequencies[start:end], width=1.5, color=colors[i], label=f'Label {i}')
        ax.legend()
    else:
        ax.bar(range(n), labels_frequencies, color = cmap(0.5))

    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency')
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title)



def find_first_occurrences(labels_array):
    """
    Returns a dictionary mapping each unique label to the index of its first occurrence in labels_array.
    """
    first_indices = {}
    unique_labels = np.unique(labels_array)
    for label in unique_labels:
        first_indices[label] = np.where(labels_array == label)[0][0]
    return first_indices



def plot_unique_frequencies_histogram(unique_frequencies, unique_counts, title="Unique Frequencies Histogram", n_bins=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    if n_bins is not None:
        unique_frequencies = unique_frequencies[:n_bins]
        unique_counts = unique_counts[:n_bins]
    plt.bar(unique_frequencies, unique_counts, width=0.008, color='tab:blue', edgecolor='black')
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.title(title)
    plt.show()





# ----------------------------------------------------------------------------



def plot_multiple_labels_frequencies_histograms(labels_frequencies_list, labels_list=None, dataset_name=None, cmap_name='inferno', title=None, y_range=None):
    """
    Plots multiple label frequency histograms vertically stacked.
    Accepts a list of arrays (possibly with different lengths).

    Args:
        labels_frequencies_list (list of np.ndarray): List of frequency arrays per hidden layer.
        labels_list (list of np.ndarray, optional): List of label arrays per hidden layer (for coloring).
        dataset_name (str, optional): Dataset name for title.
        cmap_name (str, optional): Matplotlib colormap name.
        title (str, optional): Plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k = len(labels_frequencies_list)
    fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=False)
    if k == 1:
        axes = [axes]
    cmap = plt.get_cmap(cmap_name)

    for idx in range(k):
        ax = axes[idx]
        freqs = labels_frequencies_list[idx]
        n = len(freqs)
        if y_range is not None:
            ax.set_ylim(y_range)

        if labels_list is not None:
            labels = labels_list[idx]
            first_indices = list(find_first_occurrences(labels).values())
            first_indices.append(n)
            num_segments = len(first_indices) - 1
            colors = [cmap(i / max(num_segments - 1, 1)) for i in range(num_segments)]
            for i in range(num_segments):
                start = first_indices[i]
                end = first_indices[i+1]
                ax.bar(range(start, end), freqs[start:end], width=1.5, color=colors[i], label=f'Label {i}' if idx == 0 else None)
                axes[-1].set_xlabel('Datapoint')
        else:
            ax.bar(range(n), freqs, color=cmap(0.5), width=0.5)
            axes[-1].set_xlabel('Label')
        ax.set_ylabel(f'{idx+1} hl')
        ax.set_xticks([])


    if title is not None:
        axes[0].set_title(title)
    elif dataset_name is not None:
        axes[0].set_title(f"{dataset_name}")
    plt.tight_layout()



def plot_multiple_labels_frequencies_lines(labels_frequencies_list, dataset_name=None, cmap_name='tab10', title=None):
    """
    Plots multiple label frequency arrays as line plots in the same figure.

    Args:
        labels_frequencies_list (list of np.ndarray): List of frequency arrays per hidden layer.
        dataset_name (str, optional): Dataset name for title.
        cmap_name (str, optional): Matplotlib colormap name.
        title (str, optional): Plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k = len(labels_frequencies_list)
    cmap = plt.get_cmap(cmap_name)
    plt.figure(figsize=(10, 6))

    for idx, freqs in enumerate(labels_frequencies_list):
        x = np.arange(len(freqs))
        plt.plot(x, freqs, label=f'{idx+1} hl', color=cmap(idx % cmap.N))

    plt.xlabel('Label')
    plt.ylabel('Counts')
    if title is not None:
        plt.title(title)
    elif dataset_name is not None:
        plt.title(f"{dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ...existing code...



# def plot_multiple_unique_frequencies_histograms(unique_frequencies_matrix, unique_counts_matrix, title="Unique Frequencies Histogram", n_bins=None):
#     """
#     Plots multiple unique frequencies histograms vertically stacked.

#     Args:
#         unique_frequencies_matrix (np.ndarray): Array of shape (k, n_bins).
#         unique_counts_matrix (np.ndarray): Array of shape (k, n_bins).
#         title (str, optional): Plot title.
#         n_bins (int or list, optional): Number of bins to plot, shared or per realization.
#     """
#     import matplotlib.pyplot as plt
#     k = unique_frequencies_matrix.shape[0]
#     fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=True)
#     if not isinstance(axes, np.ndarray):
#         axes = [axes]
#     for idx in range(k):
#         freqs = unique_frequencies_matrix[idx]
#         counts = unique_counts_matrix[idx]
#         bins = n_bins[idx] if isinstance(n_bins, (list, np.ndarray)) else n_bins
#         if bins is not None:
#             freqs = freqs[:bins]
#             counts = counts[:bins]
#         axes[idx].bar(freqs, counts, width=0.008, color='tab:blue', edgecolor='black')
#         axes[idx].set_ylabel(f'k={idx}')
#         axes[idx].set_yscale('log')
#         axes[idx].set_xticks([])
#     axes[-1].set_xlabel('Frequency')
#     if title is not None:
#         axes[0].set_title(title)
#     plt.tight_layout()



def plot_multiple_unique_frequencies_histograms(unique_frequencies_list, unique_counts_list, title="Unique Frequencies Histogram", n_bins=None):
    """
    Plots multiple unique frequencies histograms vertically stacked.

    Args:
        unique_frequencies_list (list of np.ndarray): Unique frequencies per realization/layer.
        unique_counts_list (list of np.ndarray): Counts per realization/layer.
        title (str, optional): Plot title.
        n_bins (int or list, optional): Number of bins to plot, shared or per realization.
    """
    import matplotlib.pyplot as plt

    k = len(unique_frequencies_list)
    fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=True)
    if k == 1:
        axes = [axes]
    for idx in range(k):

        freqs = unique_frequencies_list[idx]
        counts = unique_counts_list[idx]
        bins = n_bins[idx] if isinstance(n_bins, (list, tuple)) else n_bins
        if bins is not None:
            freqs = freqs[:bins]
            counts = counts[:bins]
        axes[idx].bar(freqs, counts, width=0.008, color='tab:blue', edgecolor='black')
        axes[idx].set_ylabel(f'{idx+1} hl')
        axes[idx].set_yscale('log')
        if idx == k-1:
            axes[idx].set_xticks(freqs)
        else:
            axes[idx].set_xticks([])
    axes[-1].set_xlabel('Frequency')
    if title is not None:
        axes[0].set_title(title)
    plt.tight_layout()




def plot_multiple_unique_frequencies_lines(unique_frequencies_list, unique_counts_list, title=None, n_bins=None, cmap_name='tab10'):
    """
    Plots multiple unique frequencies/counts arrays as line plots in the same figure.

    Args:
        unique_frequencies_list (list of np.ndarray): Unique frequencies per realization/layer.
        unique_counts_list (list of np.ndarray): Counts per realization/layer.
        title (str, optional): Plot title.
        n_bins (int or list, optional): Number of bins to plot, shared or per realization.
        cmap_name (str, optional): Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k = len(unique_frequencies_list)
    cmap = plt.get_cmap(cmap_name)
    plt.figure(figsize=(10, 6))

    for idx in range(k):
        freqs = unique_frequencies_list[idx]
        counts = unique_counts_list[idx]
        bins = n_bins[idx] if isinstance(n_bins, (list, tuple)) else n_bins
        if bins is not None:
            freqs = freqs[:bins]
            counts = counts[:bins]
        plt.plot(freqs, counts, label=f'{idx+1} hl', color=cmap(idx % cmap.N))

    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.yscale('log')
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------------------------


def compute_datapoints_labels_freq_list(dataset, ld, datapoints_array, labels_array, device, return_distances = False, save_dir=None):
    """
    Computes the frequency arrays for datapoints and labels being the closest to decoded binary latent vectors
    across multiple repetitions and hidden layer configurations.

    For each repetition and each number of hidden layers, loads the corresponding model, decodes all possible
    binary latent vectors, and computes how often each datapoint and label is selected as the closest. Optionally,
    also computes and returns the minimum distances for each decoded vector.

    Args:
        dataset (str): Name of the dataset.
        ld (int): Latent dimension.
        datapoints_array (np.ndarray): Array of shape (n_datapoints, input_dim), containing datapoints.
        labels_array (np.ndarray): Array of shape (n_datapoints,), containing labels for each datapoint.
        device (torch.device): Device to run the model on.
        return_distances (bool, optional): If True, also returns the array of minimum distances.
        save_dir (str, optional): Directory to save the computed frequency arrays.

    Returns:
        tuple:
            - np.ndarray: Array of shape (n_repetitions, n_hidden_layers, n_datapoints) with frequencies for each datapoint.
            - np.ndarray: Array of shape (n_repetitions, n_hidden_layers, n_labels) with frequencies for each label.
            - np.ndarray (optional): Array of shape (n_repetitions, n_hidden_layers, 2**latent_dim) with minimum distances for each decoded vector.
    """

    model_kwargs = {
        'input_dim': 28*28,
        'latent_dim': ld,
        'decrease_rate': 0.6,
        'device': device,
        'output_activation_encoder': nn.Sigmoid
    }
    model_path_kwargs = {
        'output_activation_encoder': 'sigmoid output',
        'train_type': 'simultaneous train',
        'latent_dim': f"{model_kwargs['latent_dim']}ld",
        'decrease_rate': '06',
        'learning_rate': '1e3',
        'train_num': 0,
    }
    model_path_kwargs['dataset'] = dataset


    repetitions = range(6)
    hidden_layers = range(1, 8)

    datapoints_freq_list = []
    labels_freq_list = []
    distances_list = []

    for repetition in repetitions:
        datapoints_freq_hl = []
        labels_freq_hl = []
        distances_hl = []

        for num_hidden_layers in hidden_layers:
            model_kwargs['hidden_layers'] = num_hidden_layers
            model_path_kwargs['num_hidden_layers'] = num_hidden_layers
            model_path_kwargs['train_num'] = repetition


            datapoints_frequencies, labels_frequencies, distances = compute_min_distances_frequencies(
                model_path_kwargs, model_kwargs, datapoints_array, labels_array, return_distances=True
            )

            datapoints_freq_hl.append(datapoints_frequencies)
            labels_freq_hl.append(labels_frequencies)
            distances_hl.append(distances)

        distances_list.append(distances_hl)
        datapoints_freq_list.append(datapoints_freq_hl)
        labels_freq_list.append(labels_freq_hl)

    repetitions_hl_datapoints_freq_array = np.array(datapoints_freq_list)
    repetitions_hl_labels_freq_array = np.array(labels_freq_list)
    repetitions_hl_distances_array = np.array(distances_list)

    if save_dir is not None:
        with open(f"{save_dir}/datapoints_freq_list_{dataset}_{ld}ld.pkl", "wb") as f:
            pickle.dump(repetitions_hl_datapoints_freq_array, f)
        with open(f"{save_dir}/labels_freq_list_{dataset}_{ld}ld.pkl", "wb") as f:
            pickle.dump(repetitions_hl_labels_freq_array, f)

    if return_distances:
        return repetitions_hl_datapoints_freq_array, repetitions_hl_labels_freq_array, repetitions_hl_distances_array
    else:
        return repetitions_hl_datapoints_freq_array, repetitions_hl_labels_freq_array



def compute_distances_list(dataset, ld, datapoints_array, device):
    """
    Computes the minimum distances between decoded binary latent vectors and their closest datapoints
    for multiple repetitions and hidden layer configurations.

    For each repetition and each number of hidden layers, loads the corresponding model, decodes all possible
    binary latent vectors, and computes the minimum Euclidean distance to the closest datapoint.

    Args:
        dataset (str): Name of the dataset.
        ld (int): Latent dimension.
        datapoints_array (np.ndarray): Array of shape (n_datapoints, input_dim), containing datapoints.
        device (torch.device): Device to run the model on.

    Returns:
        np.ndarray: Array of shape (n_repetitions, n_hidden_layers, 2**latent_dim) with minimum distances for each decoded vector.
    """

    model_kwargs = {
        'input_dim': 28*28,
        'latent_dim': ld,
        'decrease_rate': 0.6,
        'device': device,
        'output_activation_encoder': nn.Sigmoid
    }
    model_path_kwargs = {
        'output_activation_encoder': 'sigmoid output',
        'train_type': 'simultaneous train',
        'latent_dim': f"{model_kwargs['latent_dim']}ld",
        'decrease_rate': '06',
        'learning_rate': '1e3',
        'train_num': 0,
    }
    model_path_kwargs['dataset'] = dataset


    repetitions = range(6)
    hidden_layers = range(1, 8)

    distances_list = []

    for repetition in repetitions:

        distances_hl = []

        for num_hidden_layers in hidden_layers:
            model_kwargs['hidden_layers'] = num_hidden_layers
            model_path_kwargs['num_hidden_layers'] = num_hidden_layers
            model_path_kwargs['train_num'] = repetition

            distances = compute_min_distances(
                model_path_kwargs, model_kwargs, datapoints_array
            )

            distances_hl.append(distances)

        distances_list.append(distances_hl)

    repetitions_hl_distances_array = np.array(distances_list)

    return repetitions_hl_distances_array



def extract_unique_freq_counts_per_hidden_layer(hl_datapoints_freq):
    """
    For each hidden layer, extracts nonzero datapoint frequencies, corresponding labels,
    and computes unique frequencies and their counts.

    Args:
        hl_datapoints_freq (np.ndarray): Array of shape (n_hidden_layers, n_datapoints).
        labels_array (np.ndarray): Array of shape (n_datapoints,).

    Returns:
        hl_unique_frequencies (list of np.ndarray): Unique frequencies per hidden layer.
        hl_unique_counts (list of np.ndarray): Counts of unique frequencies per hidden layer.
    """

    hl_unique_frequencies = []
    hl_unique_counts = []

    for hl in range(hl_datapoints_freq.shape[0]):
        
        current_hl_nonzero_datapoints_freq = hl_datapoints_freq[hl][hl_datapoints_freq[hl] != 0]
        current_unique_frequencies, current_unique_counts = np.unique(current_hl_nonzero_datapoints_freq, return_counts=True)
        
        hl_unique_frequencies.append(current_unique_frequencies)
        hl_unique_counts.append(current_unique_counts)

    return hl_unique_frequencies, hl_unique_counts



def extract_nonzero_freq_per_hidden_layer(hl_datapoints_freq, labels_array):
    """
    For each hidden layer, extracts nonzero datapoint frequencies, corresponding labels,
    and computes unique frequencies and their counts.

    Args:
        hl_datapoints_freq (np.ndarray): Array of shape (n_hidden_layers, n_datapoints).
        labels_array (np.ndarray): Array of shape (n_datapoints,).

    Returns:
        hl_nonzero_datapoints_frequencies (list of np.ndarray): Nonzero frequencies per hidden layer.
        hl_nonzero_labels (list of np.ndarray): Corresponding labels per hidden layer.
    """
    hl_nonzero_datapoints_frequencies = []
    hl_nonzero_labels = []

    for hl in range(hl_datapoints_freq.shape[0]):
        current_hl_nonzero_datapoints_freq = hl_datapoints_freq[hl][hl_datapoints_freq[hl] != 0]
        current_hl_nonzero_labels = labels_array[hl_datapoints_freq[hl] != 0]

        hl_nonzero_datapoints_frequencies.append(current_hl_nonzero_datapoints_freq)
        hl_nonzero_labels.append(current_hl_nonzero_labels)


    return hl_nonzero_datapoints_frequencies, hl_nonzero_labels
