import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter
from AE.models import AE_0
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_model(model_kwargs, print_result=False):
    my_model = AE_0(
        model_kwargs['input_dim'],
        model_kwargs['latent_dim'],
        decrease_rate=model_kwargs['decrease_rate'],
        hidden_layers=model_kwargs['num_hidden_layers'],
        output_activation_encoder=model_kwargs['output_activation_encoder'],
        output_activation_decoder=model_kwargs['output_activation_decoder'],
        quantize_latent=model_kwargs.get('quantize_latent', False)
    ).to(model_kwargs.get('device'))
    #model_path = os.path.join(project_root, "models", "recursive", f"lm_lmb_{model_kwargs['lm_lmb']}",f"{model_kwargs['latent_dim']}ld", f"{model_kwargs['dataset']}", f"dr{model_kwargs['decrease_rate']}_{model_kwargs['num_hidden_layers']}hl_{model_kwargs['train_num']}.pth")
    #model_path = os.path.join(project_root, "models", "quantized", f"{model_kwargs['latent_dim']}ld", f"{model_kwargs['dataset']}", f"dr{model_kwargs['decrease_rate']}_{model_kwargs['num_hidden_layers']}hl_{model_kwargs['train_num']}.pth")
    model_path = os.path.join(project_root, "models", "zoomout", f"{model_kwargs['quantize_latent_path']}", f"{model_kwargs['latent_dim']}ld", f"{model_kwargs['dataset']}", f"dr{model_kwargs['decrease_rate']}_{model_kwargs['num_hidden_layers']}hl_{model_kwargs['train_num']}.pth")
    my_model.load_state_dict(torch.load(model_path, map_location=model_kwargs.get('device')))
    if print_result:
        print(f"Model loaded from {model_path}")
    return my_model



def compute_sampled_emp_states_dict(model, dataloader, num_samples=10, verbose=False, device=None):           # USED IN DEPTH.UTILS
    """
    Samples binary internal representations from the autoencoder's latent probabilities.

    Args:
        model: The autoencoder model with sigmoid activation before bottleneck
        dataloader: DataLoader containing the dataset
        num_samples: Number of samples per encoded vector
        verbose: Whether to print summary info

    Returns:
        dict: Dictionary where keys are binary state tuples and values are frequencies
    """
    model.eval()
    state_counts = defaultdict(int)
    total_samples = 0

    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            latent_vectors = model.encode(batch_data.view(batch_data.size(0), -1))  # Probabilities

            # Sample num_samples binarized vectors for each latent vector using torch.bernoulli
            for latent_vec in latent_vectors:
                probs = latent_vec.detach().cpu()
                samples = torch.bernoulli(probs.repeat(num_samples, 1))
                for sampled_vec in samples:
                    state_tuple = tuple(sampled_vec.int().numpy())
                    state_counts[state_tuple] += 1
                    total_samples += 1

    emp_states_dict = {k: v / total_samples for k, v in state_counts.items()}

    if verbose:
        print(f"Total samples processed: {total_samples}")
        print(f"Number of unique binary states found: {len(emp_states_dict)}")

    return emp_states_dict





def compute_emp_states_dict(model, dataloader, binarize_threshold=0.5, verbose=False):            # USED IN DEPTH.UTILS
    """
    Extracts binary internal representations from the autoencoder and counts their frequencies.

    Args:
        model: The autoencoder model with sigmoid activation before bottleneck
        dataloader: DataLoader containing the dataset
        device: Device to run computations on

    Returns:
        dict: Dictionary where keys are binary state tuples and values are frequencies
    """

    model.eval()
    device = model.device
    state_counts = defaultdict(int)
    total_samples = 0

    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)


            latent_vectors = model.encode(batch_data.view(batch_data.size(0), -1))      # Encode to get latent representations (after sigmoid)

            binary_states = (latent_vectors >= binarize_threshold).int()        # To binarize vectors: < 0.5 → 0, >= 0.5 → 1
            
            for i in range(binary_states.size(0)):                                      # Convert each binary vector to tuple (hashable for dictionary keys)
                state_tuple = tuple(binary_states[i].cpu().numpy())
                state_counts[state_tuple] += 1
                total_samples += 1

    emp_states_dict = {k: v / total_samples for k, v in state_counts.items()}

    if verbose:
        print(f"Total samples processed: {total_samples}")
        print(f"Number of unique binary states found: {len(emp_states_dict)}")

    return emp_states_dict





def analyze_binary_frequencies(emp_states_dict, top_k=10):
    """
    Analyze and display the most frequent binary states.

    Args:
        emp_states_dict: Dictionary from compute_emp_states_dict
        top_k: Number of top states to display
    """
    import matplotlib.pyplot as plt

    # Sort by frequency (descending)
    sorted_states = sorted(
        emp_states_dict.items(), key=lambda x: x[1], reverse=True
    )

    print(f"\nTop {top_k} most frequent binary states:")
    print("-" * 50)

    # Plot frequency distribution
    frequencies = [count for _, count in sorted_states]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(frequencies)), frequencies)
    plt.xlabel("Binary State (sorted by frequency)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Binary States")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.bar(range(min(top_k, len(frequencies))), frequencies[:top_k])
    plt.xlabel("Top Binary States")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_k} Most Frequent States")

    plt.tight_layout()
    plt.show()

    return sorted_states




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––




def mean_s_k(n, k, g):  # 0-indexed k
    """
    Calculates the expected value of the k-th feature in a set of n features under the HFM distribution.

    This function computes the mean value for the k-th (0-indexed) feature, given the total number of features and a constant parameter `g` from the HFM distribution. It handles the special case where the parameter xi equals 1 to avoid division by zero.

    Parameters
    ----------
    n : int
        Total number of features.
    k : int
        Index (0-based) of the feature for which the mean is calculated.
    g : float
        Constant parameter in the HFM distribution.

    Returns
    -------
    float
        The expected value (mean) of the k-th feature.
    """

    xi = 2 * np.exp(-g)
    if abs(xi - 1) < 1e-6:
        E = (n - (k + 1) + 2) / (2 * (n + 1))
    else:
        E = 0.5 * (1 + (xi ** (k) - 1) * (xi - 2) / (xi**n + xi - 2))
    return E




def calc_ms(state_tuple, active_category_is_zero=False):            # USED IN DEPTH.UTILS
    """
    Calculates m_s for a given state tuple, 1-indexed.
    m_s is the index of the last active neuron.
    If active_category_is_zero is True, 'active' is represented by 0, the first category.
    If no neuron is active, m_s is 0.
    """
    active_val = 0 if active_category_is_zero else 1
    for i in reversed(range(len(state_tuple))):
        if state_tuple[i] == active_val:
            return i + 1  # 1-indexed
    return 0





def calc_Z_theoretical(latent_dim, g_param):           # USED IN DEPTH.UTILS
    """
    Calculates the normalization constant Z based on the provided analytical formula.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        g_param (float): The constant 'g'.

    Returns:
        float: The normalization constant Z.
    """
    if math.isclose(g_param, math.log(2)):
        # Handles the case g = log(2) => xi = 1
        Z = 1.0 + np.exp(-g_param) * float(latent_dim)  # DIFFERENT FROM PAPER
    else:
        xi = 2.0 * math.exp(-g_param)
        if math.isclose(
            xi, 1.0
        ):  # Should be caught by g = log(2) but good for robustness
            Z = 1.0 + np.exp(-g_param) * float(latent_dim)  # DIFFERENT FROM PAPER
        else:
            sum_geometric_part = (xi**latent_dim - 1.0) / (xi - 1.0)
            Z = 1.0 + np.exp(-g_param) * sum_geometric_part  # DIFFERENT FROM PAPER
    if Z == 0:
        raise ValueError(
            "Calculated theoretical Z is zero, leading to division by zero for probabilities."
        )
    return Z





def calc_hfm_prob(m_s: float, g: float, Z: float, logits: True) -> float:
    """
    Calulates the HFM theoretical probability for a state, given m_s, g, and Z.
    If logits=True (default) it returns the log probabilities.
    """
    H_s = m_s  # max(m_s - 1, 0)
    if logits:
        return -g * H_s - np.log(Z)
    return np.exp(-g * H_s) / Z





def calc_hfm_kld(emp_states_dict, g):         # USED IN DEPTH.UTILS
    """
    Calculates the KL divergence between an empirical probability distribution
    and a theoretical distribution defined by the HFM with parameter `g`.
    Args:
        emp_states_dict (dict): A dictionary mapping states (tuples or hashable types) to their empirical probabilities.
        g (float): The parameter of the HFM model controlling the strength of the field.

    Returns:
        float: The calculated KL divergence between the empirical and theoretical distributions.
    Notes:
        - Assumes that the empirical probabilities sum to 1.
    """

    empirical_probs_values = torch.tensor(
        list(emp_states_dict.values()), dtype=torch.float32
    )
    empirical_distribution = torch.distributions.Categorical(empirical_probs_values)
    empirical_entropy = empirical_distribution.entropy()

    latent_dim = len(next(iter(emp_states_dict)))
    log_Z = math.log(calc_Z_theoretical(latent_dim, g))

    mean_H_s = 0

    for state, p_emp in emp_states_dict.items():
        m_s = calc_ms(state)  # 1-indexed
        mean_H_s += p_emp * m_s

    g_times_H_s = g * mean_H_s

    kl_div = -empirical_entropy + g_times_H_s + log_Z

    return kl_div



def calc_hfm_marginalized_prob(m_s: float, g: float, n: int) -> float:
    exp_g = np.exp(g)
    return (1 - 1/(exp_g -1)) * np.exp(-g * m_s) + (1/(exp_g - 1)) * np.exp(-g * n)

def calc_hfm_kld_with_marginalized_hfm(emp_states_dict, g):     
    
    exp_g = np.exp(g)

    empirical_probs_values = torch.tensor(
        list(emp_states_dict.values()), dtype=torch.float32
    )
    empirical_distribution = torch.distributions.Categorical(empirical_probs_values)
    empirical_entropy = empirical_distribution.entropy()

    latent_dim = len(next(iter(emp_states_dict)))

    mean_log_p_theoretical = 0
    for state, p_emp in emp_states_dict.items():
        m_s = calc_ms(state)  # 1-indexed
        mean_log_p_theoretical += p_emp * np.log(calc_hfm_marginalized_prob(m_s, g, latent_dim))

    kl_div = empirical_entropy - mean_log_p_theoretical

    return kl_div



