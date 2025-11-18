from AE.utils import compute_emp_states_dict, compute_sampled_emp_states_dict
from AE.depth_utils import flip_gauge_bits
from AE.depth_utils import compute_emp_states_dict_gauged
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader


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

