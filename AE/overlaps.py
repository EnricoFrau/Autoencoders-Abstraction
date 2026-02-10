import numpy as np
import pickle
import torch
import torch.nn as nn
import os, sys
from AE.models import AE_0
from torchvision import datasets, transforms
from AE.datasets import MNISTDigit2OnlyDataset, MNISTDigit2Dataset, FEMNISTDataset
from AE.utils import load_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr
from AE.utils import load_model

IS_TEST_MODE = False

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datasets_path = os.path.join(project_root, 'datasets')
    


def get_datapoints_labels_arrays(dataset_name, train=True):
    """
    Loads a dataset by name and returns its datapoints and labels as sorted numpy arrays.

    Args:
        dataset_name (str): Name of the dataset to load. Supported values are:
            - '2MNIST'
            - 'MNIST'
            - 'EMNIST'
            - 'FEMNIST'
        train (bool, optional): Whether to load the training set (True) or test set (False). Default is True.

    Returns:
        tuple:
            - datapoints_array (np.ndarray): Array of shape (num_samples, input_dim) containing flattened datapoints.
            - labels_array (np.ndarray): Array of shape (num_samples,) containing the corresponding labels, sorted in ascending order.

    Raises:
        ValueError: If an unknown dataset_name is provided.

    Notes:
        - The datapoints and labels are sorted by label value.
        - The function automatically downloads the dataset if not present.
    """

    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(
            datasets_path,
            train=train,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == 'EMNIST':
        dataset = datasets.EMNIST(
            datasets_path,
            split='balanced',
            train=train,
            download=True,
            transform=transforms.ToTensor()
            )
    elif dataset_name == '2MNIST':
        dataset = MNISTDigit2OnlyDataset(train=train, download=True)
    # elif dataset_name == '2MNIST':
    #     dataset = MNISTDigit2Dataset(train=train, download=True, target_size=60000)
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




def compute_rep_hl_datapoints_labels_freq(datapoints_array, model_kwargs, labels_array, repetitions_range, num_hidden_layers_range,return_distances = False, save_dir=None):
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
        model_kwargs (dict): Dictionary of model initialization parameters.
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


    # model_kwargs = {
    #     'input_dim': 28*28,
    #     'latent_dim': ld,
    #     'decrease_rate': 0.6,
    #     'device': device,
    #     'output_activation_encoder': nn.Sigmoid
    # }
    # model_path_kwargs = {
    #     'output_activation_encoder': 'sigmoid output',
    #     'train_type': 'simultaneous train',
    #     'latent_dim': f"{model_kwargs['latent_dim']}ld",
    #     'decrease_rate': '06',
    #     'learning_rate': '1e3',
    #     'train_num': 0,
    # }
    # model_path_kwargs['dataset'] = dataset


    datapoints_freq_list = []
    labels_freq_list = []
    distances_list = []

    for repetition in repetitions_range:
        model_kwargs['train_num'] = repetition
        datapoints_freq_hl = []
        labels_freq_hl = []
        distances_hl = []

        for num_hidden_layers in num_hidden_layers_range:
            model_kwargs['num_hidden_layers'] = num_hidden_layers

            datapoints_frequencies, labels_frequencies, distances = compute_min_distances_frequencies(
                model_kwargs, datapoints_array, labels_array, return_distances=True
            )

            datapoints_freq_hl.append(datapoints_frequencies)
            labels_freq_hl.append(labels_frequencies)
            distances_hl.append(distances)

        datapoints_freq_list.append(datapoints_freq_hl)
        labels_freq_list.append(labels_freq_hl)
        distances_list.append(distances_hl)

    if IS_TEST_MODE:
        print([datapoints_freq_list[i][j].shape for i in range(len(datapoints_freq_list)) for j in range(len(datapoints_freq_list[i]))])
        print([labels_freq_list[i][j].shape for i in range(len(labels_freq_list)) for j in range(len(labels_freq_list[i]))])
        print([distances_list[i][j].shape for i in range(len(distances_list)) for j in range(len(distances_list[i]))])

    repetitions_hl_datapoints_freq_array = np.array(datapoints_freq_list)
    repetitions_hl_labels_freq_array = np.array(labels_freq_list)
    repetitions_hl_distances_array = np.array(distances_list)

    if save_dir is not None:
        with open(os.path.join(f"{save_dir}/rep_hl_datapoints_freq.pkl"), "wb") as f:
            pickle.dump(repetitions_hl_datapoints_freq_array, f)
        with open(os.path.join(f"{save_dir}/rep_hl_labels_freq.pkl"), "wb") as f:
            pickle.dump(repetitions_hl_labels_freq_array, f)
        with open(os.path.join(f"{save_dir}/rep_hl_distances.pkl"), "wb") as f:
            pickle.dump(repetitions_hl_distances_array, f)

    if return_distances:
        return repetitions_hl_datapoints_freq_array, repetitions_hl_labels_freq_array, repetitions_hl_distances_array
    else:
        return repetitions_hl_datapoints_freq_array, repetitions_hl_labels_freq_array




# ----------------------------------------------------------------------------




def compute_min_distances_frequencies(model_kwargs, datapoints_array, labels_array, return_distances=False):
    """
    Computes the frequency with which each datapoint and label is the closest to a decoded binary latent vector.

    Loads the model specified by model_kwargs, decodes all possible binary latent vectors,
    finds the closest datapoint in datapoints_array for each decoded vector, and counts how often each datapoint
    and label is selected as the closest. Optionally returns the minimum distances.

    Args:
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
    decoded_binary_matrix = compute_decoded_binary_matrix(model_kwargs)
    closest_rows_indices = compute_closest_rows(decoded_binary_matrix, datapoints_array)

    if return_distances:
        distances = np.linalg.norm(decoded_binary_matrix - datapoints_array[closest_rows_indices], axis=1)

    datapoints_frequencies = compute_min_dist_datapoints_frequencies(closest_rows_indices, dataset_length=datapoints_array.shape[0])
    labels_frequencies = compute_min_dist_labels_frequencies(closest_rows_indices, labels_array)

    if return_distances:
        return datapoints_frequencies, labels_frequencies, distances
    else:
        return datapoints_frequencies, labels_frequencies
    



# ----------------------------------------------------------------------------




def compute_decoded_binary_matrix(model_kwargs):
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
    model = load_model(model_kwargs)
    model.eval()
    with torch.no_grad():
        latent_matrix = torch.tensor(binary_matrix(model.latent_dim))

        if model_kwargs.get('num_latent_samples', None) is not None:
            indices = np.random.choice(latent_matrix.shape[0], size=model_kwargs['num_latent_samples'], replace=False)
            latent_matrix = latent_matrix[indices]

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
    labels_frequencies_array = np.zeros(np.max(labels_array)+1, dtype=np.float64)
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



# ----------------------------------------------------------------------------



# def plot_multiple_labels_frequencies_histograms(labels_frequencies_list, labels_list=None, dataset_name=None, cmap_name='inferno', title=None, y_range=None):
#     """
#     Plots multiple label frequency histograms vertically stacked.
#     Accepts a list of arrays (possibly with different lengths).

#     Args:
#         labels_frequencies_list (list of np.ndarray): List of frequency arrays per hidden layer.
#         labels_list (list of np.ndarray, optional): List of label arrays per hidden layer (for coloring).
#         dataset_name (str, optional): Dataset name for title.
#         cmap_name (str, optional): Matplotlib colormap name.
#         title (str, optional): Plot title.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     k = len(labels_frequencies_list)
#     fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=False)
#     if k == 1:
#         axes = [axes]
#     cmap = plt.get_cmap(cmap_name)

#     for idx in range(k):
#         ax = axes[idx]
#         freqs = labels_frequencies_list[idx]
#         n = len(freqs)
#         if y_range is not None:
#             ax.set_ylim(y_range)

#         if labels_list is not None:
#             labels = labels_list[idx]
#             first_indices = list(find_first_occurrences(labels).values())
#             first_indices.append(n)
#             num_segments = len(first_indices) - 1

#             # Get the number of colors in the colormap
#             n_colors = cmap.N if hasattr(cmap, 'N') else 10  # fallback to 10

#             for i in range(num_segments):
#                 start = first_indices[i]
#                 end = first_indices[i+1]
#                 # Use modulo to cycle through colors
#                 color = cmap(i % n_colors / max(n_colors - 1, 1))
#                 ax.bar(range(start, end), freqs[start:end], width=1.5, color=color, label=f'Label {i}' if idx == 0 else None)
#                 axes[-1].set_xlabel('Datapoint')
#         else:
#             ax.bar(range(n), freqs, color=cmap(0.5), width=0.5)
#             axes[-1].set_xlabel('Label')
#         ax.set_ylabel(f'{idx+1} hidden layers')
#         ax.set_xticks([])


#     if title is not None:
#         axes[0].set_title(title)
#     elif dataset_name is not None:
#         axes[0].set_title(f"{dataset_name}")
#     plt.tight_layout()

def plot_multiple_labels_frequencies_histograms(
    labels_frequencies_list,
    labels_list=None,
    dataset_name=None,
    cmap_name='inferno',
    title=None,
    y_range=None,
    mark_last_n_classes=False,
    n_mark_classes=10
):
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

            n_colors = cmap.N if hasattr(cmap, 'N') else 10

            for i in range(num_segments):
                start = first_indices[i]
                end = first_indices[i+1]
                color = cmap(i % n_colors / max(n_colors - 1, 1))
                ax.bar(range(start, end), freqs[start:end], width=1.5, color=color, label=f'Label {i}' if idx == 0 else None)
                axes[-1].set_xlabel('Datapoint')

            # Mark only the start of the last n_mark_classes-th class
            if mark_last_n_classes and num_segments > n_mark_classes:
                x = first_indices[num_segments - n_mark_classes]
                ax.axvline(x, color='black', linestyle='--', linewidth=1.5)
        else:
            ax.bar(range(n), freqs, color=cmap(0.5), width=0.5)
            axes[-1].set_xlabel('Label')
        ax.set_ylabel(f'{idx+1} hidden layers')
        ax.set_xticks([])

    if title is not None:
        axes[0].set_title(title)
    elif dataset_name is not None:
        axes[0].set_title(f"{dataset_name}")
    plt.tight_layout()






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





def find_first_occurrences(labels_array):
    """
    Returns a dictionary mapping each unique label to the index of its first occurrence in labels_array.
    """
    first_indices = {}
    unique_labels = np.unique(labels_array)
    for label in unique_labels:
        first_indices[label] = np.where(labels_array == label)[0][0]
    return first_indices






def plot_multiple_distances_histograms(unique_frequencies_list, unique_counts_list, title="Unique Frequencies Histogram", x_label= None, n_bins=None, y_range=None):
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

        if y_range is not None:
            axes[idx].set_ylim(y_range)

        freqs = unique_frequencies_list[idx]
        counts = unique_counts_list[idx]
        bins = n_bins[idx] if isinstance(n_bins, (list, tuple)) else n_bins
        if bins is not None:
            freqs = freqs[:bins]
            counts = counts[:bins]
        axes[idx].bar(freqs, counts, width=1, color='darkred', edgecolor=None)
        axes[idx].set_ylabel(f'{idx+1} hl')
        axes[idx].set_xticks([])
    axes[-1].set_xlabel(x_label if x_label is not None else '')
    if title is not None:
        axes[0].set_title(title)
    plt.tight_layout()





# def plot_mean_distance_per_hidden_layer(rep_hl_distances_loaded_mean_mean, rep_hl_distances_loaded_mean_std, ld, dataset):
#     """
#     Plots the mean distance per hidden layer with error bars.

#     Args:
#         rep_hl_distances_loaded_mean_mean (np.ndarray): Mean distances per hidden layer.
#         rep_hl_distances_loaded_mean_std (np.ndarray): Standard deviation of distances per hidden layer.
#         ld (int): Latent dimension.
#         dataset (str): Dataset name.
#     """
#     import matplotlib.pyplot as plt

#     num_layers = rep_hl_distances_loaded_mean_mean.shape[0]
#     plt.figure(figsize=(8, 5))
#     plt.errorbar(
#         range(1, num_layers + 1),
#         rep_hl_distances_loaded_mean_mean,
#         yerr=rep_hl_distances_loaded_mean_std,
#         fmt='o-',                # main line: solid
#         capsize=4,
#         color='darkred',           # main line color
#         ecolor='black',          # error bar color
#         elinewidth=0.5,
#     )
#     plt.xlabel('Number of Hidden Layers')
#     plt.ylabel('Mean Distance')
#     plt.grid(False)
#     plt.show()


def plot_mean_distance_per_hidden_layer(rep_hl_distances_loaded_mean_mean, rep_hl_distances_loaded_mean_std, ld, dataset, save_dir=None):
    """
    Plots the mean distance per hidden layer with error bars, using a thin line and blue-green color style.

    Args:
        rep_hl_distances_loaded_mean_mean (np.ndarray): Mean distances per hidden layer.
        rep_hl_distances_loaded_mean_std (np.ndarray): Standard deviation of distances per hidden layer.
        ld (int): Latent dimension.
        dataset (str): Dataset name.
        save_dir (str, optional): Directory to save the plot.
    """
    import matplotlib.pyplot as plt

    num_layers = rep_hl_distances_loaded_mean_mean.shape[0]
    x = range(1, num_layers + 1)
    main_color = "#961313"   # Viridis blue-green
    error_color = "#433c3a"  # Slightly darker blue-green

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x,
        rep_hl_distances_loaded_mean_mean,
        yerr=rep_hl_distances_loaded_mean_std,
        fmt='o-',
        capsize=4,
        color=main_color,
        ecolor=error_color,
        elinewidth=0.5,
        linewidth=1,
        markersize=5,
    )

    plt.xlabel('Number of Hidden Layers', fontsize=12)
    plt.ylabel('Mean Distance', fontsize=12)
    plt.xticks(x, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/mean_distance_per_hidden_layer_{ld}ld_{dataset}.png")
    plt.show()




# def plot_entropy_or_dkltouniform(entropy_or_dkl, ylabel, title, ld, dataset, save_dir=None):
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, 8), entropy_or_dkl, marker='o')
#     plt.title(title)
#     plt.xlabel('Number of Hidden Layers')
#     plt.ylabel(ylabel)
#     plt.xticks(range(1, 8))
#     plt.grid(True)
#     if save_dir is not None:
#         plt.savefig(f"{save_dir}/{title.replace(' ', '_').lower()}_{ld}ld_{dataset}.png")
#     plt.show()

def plot_entropy_or_dkltouniform(entropy_or_dkl, ylabel, title, ld, dataset, save_dir=None, std=None):
    plt.figure(figsize=(10, 6))
    x = range(1, len(entropy_or_dkl) + 1)
    # Use a blue-green color for the main line and error bars
    main_color = '#1f9e89'  # Viridis blue-green
    error_color = "#353a3c" # Slightly darker blue-green
    if std is not None:
        plt.errorbar(
            x,
            entropy_or_dkl,
            yerr=std,
            fmt='o-',
            capsize=4,
            color=main_color,      # main line color
            ecolor=error_color,    # error bar color
            elinewidth=0.5,
        )
    else:
        plt.plot(x, entropy_or_dkl, marker='o', color=main_color)
    plt.title(title)
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{title.replace(' ', '_').lower()}_{ld}ld_{dataset}.png")
    plt.show()







# def compute_entropy_and_dkl(datapoints_freq_list_loaded):
#     """
#     Computes entropy and KL divergence to uniform for rep_hl_datapoints_freq.
#     Returns:
#         entropy_mean: mean entropy per hidden layer (shape: [num_hidden_layers])
#         dkl_uniform_mean: mean KL divergence to uniform (shape: [num_hidden_layers])
#         dkl_uniform_std: std KL divergence to uniform (shape: [num_hidden_layers])
#     """
#     rep_hl_datapoints_freq = torch.tensor(datapoints_freq_list_loaded, dtype=torch.float64)
#     rep_hl_datapoints_freq /= rep_hl_datapoints_freq.sum(dim=-1, keepdim=True)
    
#     distr_rep_hl_datapoints_freq = torch.distributions.Categorical(rep_hl_datapoints_freq)
#     entropy_mean = distr_rep_hl_datapoints_freq.entropy().mean(axis=0).numpy()
    
#     uniform_tensor = torch.full_like(rep_hl_datapoints_freq, 1.0 / rep_hl_datapoints_freq.size(-1))
#     p = rep_hl_datapoints_freq.cpu().numpy()
#     q = uniform_tensor.cpu().numpy()
#     dkl_elementwise = rel_entr(p, q)  # shape (reps, hls, datapoints)
#     dkl_uniform = dkl_elementwise.sum(axis=-1)  # shape (reps, hls)
#     dkl_uniform_mean = dkl_uniform.mean(axis=0)  # shape (hls,)
#     dkl_uniform_std = dkl_uniform.std(axis=0)    # shape (hls,)
    
#     return entropy_mean, dkl_uniform_mean, dkl_uniform_std




def compute_entropy_and_dkl(datapoints_freq_list_loaded, ld):
    """
    Computes entropy and KL divergence to uniform for rep_hl_datapoints_freq.
    Returns:
        entropy_mean: mean entropy per hidden layer (shape: [num_hidden_layers])
        dkl_uniform_mean: mean KL divergence to uniform (shape: [num_hidden_layers]), normalized by ld
        dkl_uniform_std: std KL divergence to uniform (shape: [num_hidden_layers]), normalized by ld
    """
    rep_hl_datapoints_freq = torch.tensor(datapoints_freq_list_loaded, dtype=torch.float64)
    rep_hl_datapoints_freq /= rep_hl_datapoints_freq.sum(dim=-1, keepdim=True)
    
    distr_rep_hl_datapoints_freq = torch.distributions.Categorical(rep_hl_datapoints_freq)
    entropy_mean = distr_rep_hl_datapoints_freq.entropy().mean(axis=0).numpy()
    
    uniform_tensor = torch.full_like(rep_hl_datapoints_freq, 1.0 / rep_hl_datapoints_freq.size(-1))
    p = rep_hl_datapoints_freq.cpu().numpy()
    q = uniform_tensor.cpu().numpy()
    dkl_elementwise = rel_entr(p, q)  # shape (reps, hls, datapoints)
    dkl_uniform = dkl_elementwise.sum(axis=-1)  # shape (reps, hls)
    dkl_uniform_mean = dkl_uniform.mean(axis=0) / ld  # normalize by ld
    dkl_uniform_std = dkl_uniform.std(axis=0) / ld    # normalize by ld
    
    return entropy_mean, dkl_uniform_mean, dkl_uniform_std



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------












# ----------------------------------------------------------------------------



def plot_overlap_heatmap(overlap_matrix, dataset_names=None, title="Overlap Matrix Heatmap", cmap="magma", vmin=0, vmax=12, ax=None):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(overlap_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Overlap Value")
    if dataset_names is not None:
        ax.set_xticks(np.arange(len(dataset_names)))
        ax.set_xticklabels(dataset_names, rotation=45)
        ax.set_yticks(np.arange(len(dataset_names)))
        ax.set_yticklabels(dataset_names)
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    plt.tight_layout()



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



def compute_overlap_matrix_for_repetitions(datasets, model_path_kwargs, model_kwargs, repetitions):
    distances_mean_matrices = []
    distances_std_matrices = []

    for rep_i in repetitions:
        for rep_j in repetitions:

            distances_mean_matrix, distances_std_matrix = compute_overlap_matrix(datasets, model_path_kwargs, model_kwargs, repetitions=(rep_i, rep_j))
            distances_mean_matrices.append(distances_mean_matrix)
            distances_std_matrices.append(distances_std_matrix)

    mean_of_distances_means = np.mean(distances_mean_matrices, axis=0)
    mean_of_distances_stds = np.mean(distances_std_matrices, axis=0)
    
    return mean_of_distances_means, mean_of_distances_stds




# ----------------------------------------------------------------------------




def compute_overlap_matrix(datasets, model_path_kwargs, model_kwargs, repetitions=(0, 0)):

    distances_mean_matrix = np.zeros((len(datasets), len(datasets)))
    distances_std_matrix = np.zeros((len(datasets), len(datasets)))

    for i, dataset_i in enumerate(datasets):
        for j, dataset_j in enumerate(datasets):
            
            # if dataset_i == dataset_j:
            #     continue

            dataset_pair = (dataset_i, dataset_j)

            distances = compute_all_decoded_features_distances_without_repetitions(
                    model_path_kwargs=model_path_kwargs,
                    model_kwargs=model_kwargs,
                    repetitions=repetitions,
                    datasets=dataset_pair
                )

            distances_mean = distances.mean()
            distances_std = distances.std()

            distances_mean_matrix[i, j] = distances_mean
            distances_std_matrix[i, j] = distances_std

    return distances_mean_matrix, distances_std_matrix



# ----------------------------------------------------------------------------



def compute_all_decoded_features_distances_without_repetitions(model_kwargs, repetitions = (0,0), datasets=None):
    model_kwargs['dataset'] = datasets[0]
    model_kwargs['train_num'] = repetitions[0]
    model_D = load_model(model_kwargs)

    model_path_kwargs['dataset'] = datasets[1]
    model_path_kwargs['train_num'] = repetitions[1]
    model_D_prime = load_model(model_path_kwargs, model_kwargs)

    distances = compute_all_decoded_features_distances(
        compute_all_decoded_features(model_D), 
        compute_all_decoded_features(model_D_prime)
    )
    return distances



# ---------------------------------------------------------------------------- 



def compute_all_decoded_features_distances(all_features_D, all_features_D_prime):
    distances = np.zeros(len(all_features_D))
    for i in range(len(all_features_D)):
            distances[i] = calc_euclidean_distance(all_features_D[i], all_features_D_prime[i])
    return distances




# ---------------------------------------------------------------------------



def compute_all_decoded_features(model):
    all_features = []
    for i in range(model.latent_dim):
        feature_i = get_feature_i(i, model)
        all_features.append(feature_i)
    return all_features



def calc_euclidean_distance(feature_i_D, feature_i_D_prime):
    return np.linalg.norm(feature_i_D - feature_i_D_prime)



# ---------------------------------------------------------------------------




def get_feature_i(i, model):
    model.eval()
    with torch.no_grad():
        ld = model.latent_dim
        input_feature_state = torch.zeros((1, ld))
        input_feature_state[0, i] = 1
        feature_i = model.decode(input_feature_state).cpu().numpy()
        feature_i = feature_i.squeeze(0)  # Remove the first dimension
    return feature_i



# ----------------------------------------------------------------------------



# def compute_all_decoded_features_dist_over_repetitions(model_path_kwargs, model_kwargs, repetitions=range(6), datasets=None):

#     distances_over_repetitions = np.array([])

#     for i in repetitions:
#         model_path_kwargs['dataset'] = datasets[0]
#         model_path_kwargs['train_num'] = i
#         model_D = load_model(model_path_kwargs, model_kwargs)

#         for j in repetitions:
#             model_path_kwargs['dataset'] = datasets[1]
#             model_path_kwargs['train_num'] = j
#             model_D_prime = load_model(model_path_kwargs, model_kwargs)

#             distances = compute_all_decoded_features_distances(
#                 compute_all_decoded_features(model_D), 
#                 compute_all_decoded_features(model_D_prime)
#             )

#             distances_over_repetitions = np.append(distances_over_repetitions, distances)

#     return distances_over_repetitions













# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------





















#----------------------------------------------------------------------------










# def plot_labels_frequencies_histogram(labels_frequencies, first_indices=None, title=None, cmap_name='inferno', ax=None):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     if ax is None:
#         ax = plt.gca()

#     n = len(labels_frequencies)
#     cmap = plt.get_cmap(cmap_name)

#     if first_indices is not None:
#         indices = [int(x) for x in first_indices]
#         indices.append(n)
#         num_segments = len(indices) - 1
#         colors = [cmap(i / max(num_segments - 1, 1)) for i in range(num_segments)]
#         for i in range(num_segments):
#             start = indices[i]
#             end = indices[i+1]
#             ax.bar(range(start, end), labels_frequencies[start:end], width=1.5, color=colors[i], label=f'Label {i}')
#         ax.legend()
#     else:
#         ax.bar(range(n), labels_frequencies, color = cmap(0.5))

#     ax.set_xlabel('Label')
#     ax.set_ylabel('Frequency')
#     ax.set_xticks([])
#     if title is not None:
#         ax.set_title(title)






# def plot_unique_frequencies_histogram(unique_frequencies, unique_counts, title="Unique Frequencies Histogram", n_bins=None):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 5))
#     if n_bins is not None:
#         unique_frequencies = unique_frequencies[:n_bins]
#         unique_counts = unique_counts[:n_bins]
#     plt.bar(unique_frequencies, unique_counts, width=0.008, color='tab:blue', edgecolor='black')
#     plt.xlabel('Frequency')
#     plt.ylabel('Count')
#     plt.yscale('log')
#     plt.title(title)
#     plt.show()





# ----------------------------------------------------------------------------






# def plot_multiple_labels_frequencies_lines(labels_frequencies_list, dataset_name=None, cmap_name='tab10', title=None):
#     """
#     Plots multiple label frequency arrays as line plots in the same figure.

#     Args:
#         labels_frequencies_list (list of np.ndarray): List of frequency arrays per hidden layer.
#         dataset_name (str, optional): Dataset name for title.
#         cmap_name (str, optional): Matplotlib colormap name.
#         title (str, optional): Plot title.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     k = len(labels_frequencies_list)
#     cmap = plt.get_cmap(cmap_name)
#     plt.figure(figsize=(10, 6))

#     for idx, freqs in enumerate(labels_frequencies_list):
#         x = np.arange(len(freqs))
#         plt.plot(x, freqs, label=f'{idx+1} hl', color=cmap(idx % cmap.N))

#     plt.xlabel('Label')
#     plt.ylabel('Counts')
#     if title is not None:
#         plt.title(title)
#     elif dataset_name is not None:
#         plt.title(f"{dataset_name}")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


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



# def plot_multiple_unique_frequencies_histograms(unique_frequencies_list, unique_counts_list, title="Unique Frequencies Histogram", n_bins=None):
#     """
#     Plots multiple unique frequencies histograms vertically stacked.

#     Args:
#         unique_frequencies_list (list of np.ndarray): Unique frequencies per realization/layer.
#         unique_counts_list (list of np.ndarray): Counts per realization/layer.
#         title (str, optional): Plot title.
#         n_bins (int or list, optional): Number of bins to plot, shared or per realization.
#     """
#     import matplotlib.pyplot as plt

#     k = len(unique_frequencies_list)
#     fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=True)
#     if k == 1:
#         axes = [axes]
#     for idx in range(k):

#         freqs = unique_frequencies_list[idx]
#         counts = unique_counts_list[idx]
#         bins = n_bins[idx] if isinstance(n_bins, (list, tuple)) else n_bins
#         if bins is not None:
#             freqs = freqs[:bins]
#             counts = counts[:bins]
#         axes[idx].bar(freqs, counts, width=0.008, color='tab:blue', edgecolor='black')
#         axes[idx].set_ylabel(f'{idx+1} hl')
#         axes[idx].set_yscale('log')
#         if idx == k-1:
#             axes[idx].set_xticks(freqs)
#         else:
#             axes[idx].set_xticks([])
#     axes[-1].set_xlabel('Frequency')
#     if title is not None:
#         axes[0].set_title(title)
#     plt.tight_layout()




# def plot_multiple_unique_frequencies_lines(unique_frequencies_list, unique_counts_list, title=None, n_bins=None, cmap_name='tab10'):
#     """
#     Plots multiple unique frequencies/counts arrays as line plots in the same figure.

#     Args:
#         unique_frequencies_list (list of np.ndarray): Unique frequencies per realization/layer.
#         unique_counts_list (list of np.ndarray): Counts per realization/layer.
#         title (str, optional): Plot title.
#         n_bins (int or list, optional): Number of bins to plot, shared or per realization.
#         cmap_name (str, optional): Matplotlib colormap name.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     k = len(unique_frequencies_list)
#     cmap = plt.get_cmap(cmap_name)
#     plt.figure(figsize=(10, 6))

#     for idx in range(k):
#         freqs = unique_frequencies_list[idx]
#         counts = unique_counts_list[idx]
#         bins = n_bins[idx] if isinstance(n_bins, (list, tuple)) else n_bins
#         if bins is not None:
#             freqs = freqs[:bins]
#             counts = counts[:bins]
#         plt.plot(freqs, counts, label=f'{idx+1} hl', color=cmap(idx % cmap.N))

#     plt.xlabel('Frequency')
#     plt.ylabel('Count')
#     plt.yscale('log')
#     if title is not None:
#         plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()



# ----------------------------------------------------------------------------




# def compute_distances_list(dataset, ld, datapoints_array, device):
#     """
#     Computes the minimum distances between decoded binary latent vectors and their closest datapoints
#     for multiple repetitions and hidden layer configurations.

#     For each repetition and each number of hidden layers, loads the corresponding model, decodes all possible
#     binary latent vectors, and computes the minimum Euclidean distance to the closest datapoint.

#     Args:
#         dataset (str): Name of the dataset.
#         ld (int): Latent dimension.
#         datapoints_array (np.ndarray): Array of shape (n_datapoints, input_dim), containing datapoints.
#         device (torch.device): Device to run the model on.

#     Returns:
#         np.ndarray: Array of shape (n_repetitions, n_hidden_layers, 2**latent_dim) with minimum distances for each decoded vector.
#     """

#     model_kwargs = {
#         'input_dim': 28*28,
#         'latent_dim': ld,
#         'decrease_rate': 0.6,
#         'device': device,
#         'output_activation_encoder': nn.Sigmoid
#     }
#     model_path_kwargs = {
#         'output_activation_encoder': 'sigmoid output',
#         'train_type': 'simultaneous train',
#         'latent_dim': f"{model_kwargs['latent_dim']}ld",
#         'decrease_rate': '06',
#         'learning_rate': '1e3',
#         'train_num': 0,
#     }
#     model_path_kwargs['dataset'] = dataset


#     repetitions = range(6)
#     hidden_layers = range(1, 8)

#     distances_list = []

#     for repetition in repetitions:

#         distances_hl = []

#         for num_hidden_layers in hidden_layers:
#             model_kwargs['hidden_layers'] = num_hidden_layers
#             model_path_kwargs['num_hidden_layers'] = num_hidden_layers
#             model_path_kwargs['train_num'] = repetition

#             distances = compute_min_distances(
#                 model_path_kwargs, model_kwargs, datapoints_array
#             )

#             distances_hl.append(distances)

#         distances_list.append(distances_hl)

#     repetitions_hl_distances_array = np.array(distances_list)

#     return repetitions_hl_distances_array



# def extract_unique_freq_counts_per_hidden_layer(hl_datapoints_freq):
#     """
#     For each hidden layer, extracts nonzero datapoint frequencies, corresponding labels,
#     and computes unique frequencies and their counts.

#     Args:
#         hl_datapoints_freq (np.ndarray): Array of shape (n_hidden_layers, n_datapoints).
#         labels_array (np.ndarray): Array of shape (n_datapoints,).

#     Returns:
#         hl_unique_frequencies (list of np.ndarray): Unique frequencies per hidden layer.
#         hl_unique_counts (list of np.ndarray): Counts of unique frequencies per hidden layer.
#     """

#     hl_unique_frequencies = []
#     hl_unique_counts = []

#     for hl in range(hl_datapoints_freq.shape[0]):
        
#         current_hl_nonzero_datapoints_freq = hl_datapoints_freq[hl][hl_datapoints_freq[hl] != 0]
#         current_unique_frequencies, current_unique_counts = np.unique(current_hl_nonzero_datapoints_freq, return_counts=True)
        
#         hl_unique_frequencies.append(current_unique_frequencies)
#         hl_unique_counts.append(current_unique_counts)

#     return hl_unique_frequencies, hl_unique_counts














