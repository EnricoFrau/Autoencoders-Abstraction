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



def plot_multiple_labels_frequencies_histograms(labels_frequencies_matrix, first_indices=None, title=None, cmap_name='inferno', ax=None):
    """
    Plots multiple label frequency histograms vertically stacked.

    Args:
        labels_frequencies_matrix (np.ndarray): Array of shape (k, n_labels).
        first_indices (list or list of lists, optional): Indices for segment coloring, either shared or per realization.
        title (str, optional): Plot title.
        cmap_name (str, optional): Matplotlib colormap name.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k, n = labels_frequencies_matrix.shape
    cmap = plt.get_cmap(cmap_name)

    if ax is None:
        fig, axes = plt.subplots(k, 1, figsize=(8, 2*k), sharex=True)
    else:
        axes = [ax] * k

    for idx in range(k):
        ax_i = axes[idx]
        freqs = labels_frequencies_matrix[idx]
        # Handle first_indices per realization
        if first_indices is not None:
            if isinstance(first_indices[0], (list, np.ndarray)):
                indices = [int(x) for x in first_indices[idx]]
            else:
                indices = [int(x) for x in first_indices]
            indices.append(n)
            num_segments = len(indices) - 1
            colors = [cmap(i / max(num_segments - 1, 1)) for i in range(num_segments)]
            for i in range(num_segments):
                start = indices[i]
                end = indices[i+1]
                ax_i.bar(range(start, end), freqs[start:end], width=1.5, color=colors[i], label=f'Label {i}' if idx == 0 else None)
            if idx == 0:
                ax_i.legend()
        else:
            ax_i.bar(range(n), freqs, color=cmap(0.5))
        ax_i.set_ylabel(f'k={idx}')
        ax_i.set_xticks([])

    axes[-1].set_xlabel('Label')
    if title is not None:
        axes[0].set_title(title)
    plt.tight_layout()



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
        axes[idx].set_ylabel(f'k={idx}')
        axes[idx].set_yscale('log')
        axes[idx].set_xticks([])
    axes[-1].set_xlabel('Frequency')
    if title is not None:
        axes[0].set_title(title)
    plt.tight_layout()
