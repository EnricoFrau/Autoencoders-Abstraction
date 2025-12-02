from AE.utils import compute_emp_states_dict, compute_sampled_emp_states_dict
from AE.depth_utils import flip_gauge_bits
from AE.depth_utils import compute_emp_states_dict_gauged
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_bottleneck_neurons_activ_freq(
        model: Module,
        dataloader: DataLoader,
        binarize_threshold = 0.5,
        flip_gauge = False,
        device = None,
):          
    """
    Calculates the activation frequencies of bottleneck neurons in an autoencoder model.
    This function computes how frequently each neuron in the bottleneck layer is activated
    across the dataset provided by the dataloader. The activations are binarized using the
    specified threshold before calculating the frequencies.
    Args:
        model (torch.nn.Module): The autoencoder model containing the bottleneck layer.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        binarize_threshold (float, optional): Threshold to binarize neuron activations.
            Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
            Defaults to false.
    Returns:
        np.ndarray: A numpy array with activation frequency as entries.
        (i.e., the proportion of samples for which the neuron is active).
    """
    
    if binarize_threshold is None:
        emp_states_dict = compute_sampled_emp_states_dict(model, dataloader, device=device)
    else:
        emp_states_dict = compute_emp_states_dict(model, dataloader, binarize_threshold)

    if flip_gauge:
        emp_states_dict = flip_gauge_bits(emp_states_dict)
    
    bottleneck_activ_freq = calc_neurons_activ_freq(emp_states_dict)

    return bottleneck_activ_freq




def compute_bottleneck_neurons_activ_freq_gauged(
        model: Module,
        dataloader: DataLoader,
        binarize_threshold = 0.5,
):          
    """
    Calculates the activation frequencies of bottleneck neurons in an autoencoder model, after finding the correct gauge.
    This function computes how frequently each neuron in the bottleneck layer is activated
    across the dataset provided by the dataloader. The activations are binarized using the
    specified threshold before calculating the frequencies.
    Args:
        model (torch.nn.Module): The autoencoder model containing the bottleneck layer.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        binarize_threshold (float, optional): Threshold to binarize neuron activations.
            Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
            Defaults to false.
    Returns:
        np.ndarray: A numpy array with activation frequency as entries.
        (i.e., the proportion of samples for which the neuron is active).
    """
    
    emp_states_dict_gauged = compute_emp_states_dict_gauged(model, dataloader, binarize_threshold)
    
    bottleneck_activ_freq = calc_neurons_activ_freq(emp_states_dict_gauged)

    return bottleneck_activ_freq




def calc_neurons_activ_freq(emp_states_dict):       # USED IN compute_bottleneck_neurons_activ_freq
    """
    Given a dictionary where keys are binary tuples (or lists) and values are frequencies,
    returns a vector with the weighted sum (activation frequency) for each neuron/dimension.

    Args:
        emp_states_dict (dict): {(0,1,1,...): freq, ...}

    Returns:
        np.ndarray: activation frequency for each dimension
    """
    # Convert keys to numpy array
    states = np.array(list(emp_states_dict.keys()))
    freqs = np.array(list(emp_states_dict.values()))
    # Weighted sum along each column (dimension)
    activation_freqs = np.average(states, axis=0, weights=freqs)
    return activation_freqs



#=========================================================================================================




def get_encoded_data(model, dataloader, device=None):
    """
    Returns a tensor of shape (n_data, latent_dim) containing the encoded data from the model.
    
    Args:
        model (torch.nn.Module): The autoencoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        device (torch.device, optional): Device to run the model on.
    
    Returns:
        torch.Tensor: Encoded data of shape (n_data, latent_dim).
    """
    model.eval()
    encoded_list = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # In case dataset returns (data, label)
            if device is not None:
                batch = batch.to(device)
            encoded = model.encode(batch)
            encoded_list.append(encoded.cpu())
    return torch.cat(encoded_list, dim=0)



def compute_reconstruction_distances(model, dataloader, device):
    model.eval()
    distances = []
    input_dim = model.input_dim

    with torch.no_grad():
        for batch in dataloader:
            # If batch is a tuple (inputs, labels), take only inputs
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            # Flatten input if necessary
            if inputs.dim() > 2:
                inputs_flat = inputs.view(inputs.size(0), -1)
            else:
                inputs_flat = inputs

            outputs = model(inputs_flat)

            # If recursive_last_layer, slice output to input_dim
            if getattr(model, 'recursive_last_layer', False):
                outputs = outputs[..., :input_dim]

            outputs_flat = outputs.view(outputs.size(0), -1)

            batch_distances = torch.nn.functional.mse_loss(outputs_flat, inputs_flat, reduction='none').mean(dim=1)
            distances.append(batch_distances.cpu())

    return torch.cat(distances)


def compute_bottleneck_vs_decoded_latent_distances(model, dataloader, device):
    """
    Computes the MSE distance between the bottleneck (encoded) representation and
    the last latent_dim neurons of the decoded output for each datapoint.
    Only works for AE_0 with recursive_last_layer=True.
    Returns a tensor of distances of length n_datapoints.
    """
    model.eval()
    distances = []
    input_dim = model.input_dim
    latent_dim = model.latent_dim

    with torch.no_grad():
        for batch in dataloader:
            # If batch is a tuple (inputs, labels), take only inputs
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            # Flatten input if necessary
            if inputs.dim() > 2:
                inputs_flat = inputs.view(inputs.size(0), -1)
            else:
                inputs_flat = inputs

            # Get bottleneck representation
            encoded = model.encode(inputs_flat)
            # Get decoded output
            decoded = model.decode(encoded)

            # Take last latent_dim neurons from decoded output
            decoded_latent = decoded[..., input_dim:input_dim+latent_dim]

            # Flatten if necessary
            encoded_flat = encoded.view(encoded.size(0), -1)
            decoded_latent_flat = decoded_latent.view(decoded_latent.size(0), -1)

            # Compute MSE for each sample in batch
            batch_distances = F.mse_loss(decoded_latent_flat, encoded_flat, reduction='none').mean(dim=1)
            distances.append(batch_distances.cpu())

    return torch.cat(distances)



def plot_distance_histogram(distances, bins=50, title="Reconstruction Distance Histogram", xlim=None, ylim=None):
    """
    Plots a histogram of the reconstruction distances.
    
    Args:
        distances (Tensor or array-like): 1D tensor or array of distances.
        bins (int): Number of histogram bins.
        title (str): Plot title.
    """
    distances_np = distances.cpu().numpy() if hasattr(distances, "cpu") else distances
    plt.figure(figsize=(8, 5))
    plt.hist(distances_np, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel("Reconstruction Distance (MSE)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
