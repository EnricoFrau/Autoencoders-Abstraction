import numpy as np
from itertools import permutations, combinations
import os
import matplotlib.pyplot as plt
import random
import math
from torch.nn import Module
from torch.utils.data import DataLoader
import torch

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from AE.utils import calc_hfm_kld, calc_hfm_kld_with_marginalized_hfm

from AE.utils import calc_Z_theoretical
from AE.utils import calc_ms
from AE.overlaps import load_model
from AE.overlaps import load_model
from AE.utils import compute_emp_states_dict, compute_sampled_emp_states_dict

IS_TEST_MODE = False


# –––––––––––––––––––––––––––––––––––– EXPORTED TO DEPTH_ANALYSIS –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def mean_over_outer_dict(dataset_dicts, selected_train_nums=None):
    # If not specified, use all train_nums
    if selected_train_nums is None:
        train_nums = list(dataset_dicts.keys())
    else:
        train_nums = list(selected_train_nums)
    # Get all dataset names (assume all inner dicts have the same keys)
    dataset_names = list(dataset_dicts[train_nums[0]].keys())
    # Prepare output
    mean_dict = {ds: [] for ds in dataset_names}
    # Find the minimum length for each dataset to avoid shape mismatch
    min_lengths = {ds: min(len(dataset_dicts[tn][ds]) for tn in train_nums) for ds in dataset_names}
    # Compute mean for each dataset and each position
    for ds in dataset_names:
        # Stack lists for this dataset across all selected train_nums, up to min length
        stacked = np.array([dataset_dicts[tn][ds][:min_lengths[ds]] for tn in train_nums])
        mean_dict[ds] = stacked.mean(axis=0).tolist()
    return mean_dict



def write_encoded_dataset_on_file_sigmoid_output(data_loader, model_kwargs, device, model_path_kwargs, num_hidden_layers_range):
    """
    Encodes a dataset using autoencoder models with varying numbers of hidden layers and writes the encoded outputs to text files.

    For each value in `num_hidden_layers_range`, this function:
        - Loads the corresponding autoencoder model.
        - Encodes the input data from `data_loader` using the model's encoder.
        - Writes each encoded vector to a text file, with values formatted as tuples and separated by commas.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader providing the input data to encode.
        model_kwargs (dict): Keyword arguments for constructing the AE_0 model.
        device (torch.device): Device on which to run the model (e.g., 'cpu' or 'cuda').
        model_path_kwargs (dict): Dictionary containing parameters for constructing the model path and output file path.
        num_hidden_layers_range (iterable): Iterable of integers specifying the number of hidden layers to use for each model.

    Output:
        For each number of hidden layers, a text file is created containing the encoded dataset, with one encoded vector per line.
    """
    
    for num_hidden_layers in num_hidden_layers_range:

        model_path_kwargs['num_hidden_layers'] = num_hidden_layers
        model_kwargs['hidden_layers'] = num_hidden_layers

        my_model = load_model(model_path_kwargs, model_kwargs)

        save_dir = f"../pure encoding/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['train_type']}/sigmoid output decoder/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/{model_path_kwargs['train_num']}/{num_hidden_layers}hl.txt"
        my_model.eval()
        with open(save_dir, 'w') as f:
            with torch.no_grad():
                for batch in data_loader:
                    # If batch is (data, label), take only data
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch
                    data = data.to(device)
                    encoded = my_model.encode(data)
                    # Flatten each encoded vector and write as space-separated values
                    for vec in encoded.cpu().numpy():
                        formatted = '(' + ', '.join(f"{v:.8f}" for v in vec.tolist()) + ')'
                        f.write(formatted + '\n')




def compute_dataset_klds_gs_dict_from_sampled_binarized_vectors_(dataset, data_loader, model_kwargs, device, num_hidden_layers_range, hfm_distribution = 'pure', dataset_klds_dict = None, dataset_gs_dict = None, save_permutations = False):
    """
    Computes and stores the Kullback-Leibler divergences (KLDs) and optimal 'g' values for a given dataset
    using autoencoder models with varying numbers of hidden layers.

    For each value in `num_hidden_layers_range`, this function:
        - Loads the corresponding autoencoder model.
        - Computes the KLD and optimal 'g' value using sampled binarized latent vectors.
        - Appends the results to the provided or newly created dictionaries for the specified dataset.

    Args:
        dataset (str): Name of the dataset (e.g., 'MNIST', 'EMNIST', '2MNIST').
        data_loader (torch.utils.data.DataLoader): DataLoader providing the input data.
        model_kwargs (dict): Keyword arguments for constructing the AE_0 model.
        device (torch.device): Device on which to run the model (e.g., 'cpu' or 'cuda').
        model_path_kwargs (dict): Dictionary containing parameters for constructing the model path.
        num_hidden_layers_range (iterable): Iterable of integers specifying the number of hidden layers to use for each model.
        dataset_klds_dict (dict, optional): Dictionary to store KLDs for each dataset. If None, a new one is created.
        dataset_gs_dict (dict, optional): Dictionary to store 'g' values for each dataset. If None, a new one is created.

    Returns:
        tuple: (dataset_klds_dict, dataset_gs_dict) containing the computed KLDs and 'g' values for the given dataset.
    """

    if dataset_klds_dict is None:
        dataset_klds_dict = {
            '2MNIST': [],
            'MNIST': [],
            'EMNIST': []}
    if dataset_gs_dict is None:
        dataset_gs_dict = {
            '2MNIST': [],
            'MNIST': [],
            'EMNIST': []}


    for num_hidden_layers in num_hidden_layers_range:

        if IS_TEST_MODE:
            print(f"{num_hidden_layers} hidden layers...")

        model_kwargs['num_hidden_layers'] = num_hidden_layers
        model = load_model(model_kwargs, device=device)
        model.eval()

        kld, g = calc_hfm_kld_with_optimal_g(model, data_loader, return_g=True, binarize_threshold=None, hfm_distribution=hfm_distribution, save_permutations=save_permutations, model_kwargs=model_kwargs)

        dataset_klds_dict[dataset].append(kld)
        dataset_gs_dict[dataset].append(g)

        #save_dir = f"../pure encoding/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['train_type']}/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/{model_path_kwargs['train_num']}/{num_hidden_layers}hl.txt"
        #save_dir = f"../Images/relu output/simultaneous train/{latent_dim}features/"
        save_dir = None

    return dataset_klds_dict, dataset_gs_dict




# –––––––––––––––––––––––––––––––––– EXPORTED TO DETPH_ANALYSIS ––––––––––––––––––––––––––––––––––––––




def calc_hfm_kld_with_optimal_g(            # used in compute_klds_gs_lst_with_fixed_threshold AND EXPORTED TO DEPTH_ANALYSIS
        model: Module,
        data_loader: DataLoader,
        return_g = False,
        binarize_threshold = 0.5,
        hfm_distribution = 'pure',
        save_permutations = False,
        model_kwargs = {}):   
    """
    Calculates the KL divergence between the empirical latent state distribution (from the model and data_loader)
    and the HFM model with its optimal parameter g (chosen to best match the empirical mean).

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data.
        return_g (bool, optional): If True, also returns the optimal g value. Defaults to False.
        binarize_threshold (float, optional): Threshold for binarizing latent activations. Defaults to 0.5.

    Returns:
        float: KL divergence between empirical states and HFM with optimal g.
        float (optional): The optimal g value (if return_g is True).
    """

    if IS_TEST_MODE:
        print(f"Calculating emp_states_dict_gauged")

    if save_permutations:
        emp_states_dict_gauged, gauge_perm = compute_emp_states_dict_gauged(model, data_loader, binarize_threshold=binarize_threshold, hfm_distribution=hfm_distribution, return_perm=True, model_kwargs=model_kwargs)
        perm_save_dir = os.path.join(project_root, "gauges", "permutations", model_kwargs['output_activation_encoder_path'], model_kwargs['output_activation_decoder_path'],str(model_kwargs['latent_dim']), model_kwargs['dataset'], model_kwargs['lm_lmb'], ++f"perm_{model_kwargs['num_hidden_layers']}hl.txt")
        os.makedirs(os.path.dirname(perm_save_dir), exist_ok=True)
        train_num = model_kwargs.get('train_num', 'unknown')
        with open(perm_save_dir, 'a') as f:
            f.write(f"{train_num}\t{gauge_perm}\n")
    else:
        emp_states_dict_gauged = compute_emp_states_dict_gauged(model, data_loader, binarize_threshold=binarize_threshold, hfm_distribution=hfm_distribution)

    if IS_TEST_MODE:
        print(f"Calculating optimal g and KL divergence")
    if hfm_distribution == 'pure':
        optimal_g = calc_optimal_g(emp_states_dict_gauged)
        kl_div = calc_hfm_kld(emp_states_dict_gauged, optimal_g)
    elif hfm_distribution == 'marginalized':
        optimal_g = calc_optimal_g_with_marginalized_hfm(emp_states_dict_gauged)
        if IS_TEST_MODE:
            print(f"optimal_g: {optimal_g}")
            print("Calculating KL divergence with marginalized HFM")
        kl_div = calc_hfm_kld_with_marginalized_hfm(emp_states_dict_gauged, optimal_g)
        if IS_TEST_MODE:
            print(f"kl_div: {kl_div}")
    else:
        raise ValueError("hfm_distribution must be either 'pure' or 'marginalized'")

    return kl_div, optimal_g if return_g else kl_div



#–––––––––––––––––––––––––––––––––––––––– USED IN THE FUNCTIONS DEFINED ABOVE - 1 ––––––––––––––––––––––––––––––––––––––––––––––––––




def compute_emp_states_dict_gauged(                 # USED IN calc_hfm_kld_with_optimal_g AND EXPORTED TO DEPTH.UTILS
        model: Module,
        data_loader: DataLoader,
        binarize_threshold = 0.5,
        flip_gauge = True,
        hfm_distribution = 'pure',
        brute_force = False,
        return_perm = False,
        verbose = False,
        model_kwargs = None
    ):
    """
    Computes the empirical latent state distribution from a model and dataloader, applies gauge flipping
    (bit inversion based on the most frequent state), and finds the permutation of latent dimensions
    that minimizes the KL divergence with the HFM model. The permutation is found using either brute-force
    search or simulated annealing.

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data.
        binarize_threshold (float, optional): Threshold for binarizing latent activations. Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
        brute_force (bool, optional): If True, uses brute-force search for optimal permutation; otherwise uses simulated annealing. Defaults to False.
        return_perm (bool, optional): If True, returns both the gauged state dictionary and the optimal permutation. Defaults to False.
        verbose (bool, optional): If True, prints progress information during permutation search. Defaults to False.

    Returns:
        dict: Empirical state dictionary with columns permuted and gauge-flipped to minimize KL divergence with the HFM model.
        tuple (optional): The optimal permutation of latent dimensions (if return_perm is True).
    """
    
    if IS_TEST_MODE:
        print(f"Computing emp_states_dict")    
    if binarize_threshold is None:
        emp_states_dict = compute_sampled_emp_states_dict(model, data_loader, num_samples=5, device=model_kwargs.get('device', None))
    else:
        emp_states_dict = compute_emp_states_dict(model, data_loader, binarize_threshold)

    if flip_gauge:
        emp_states_dict = flip_gauge_bits(emp_states_dict, save_flip_gauge=return_perm, model_kwargs=model_kwargs)

    if brute_force:
        gauge_perm = compute_perm_minimizing_hfm_kld_brute_force(emp_states_dict, verbose=verbose)
    else:
        gauge_perm = compute_perm_minimizing_hfm_kld_simul_anneal(emp_states_dict, hfm_distribution = hfm_distribution, g=0.8, verbose=verbose)

    emp_states_dict_gauged = {tuple(k[i] for i in gauge_perm): v for k, v in emp_states_dict.items()}

    if return_perm:
        return emp_states_dict_gauged, gauge_perm
    else:
        return emp_states_dict_gauged



def calc_optimal_g(                                 # USED IN calc_hfm_kld_with_optimal_g
        emp_states_dict_gauged,
        plot_graph = False,
        verbose = False
):
    """
    Finds the optimal value of the parameter 'g' for the HFM such that
    the theoretical mean of the expected value (m_s average) matches the empirical mean (m_s mean)
    computed from the provided gauged states.

    The function computes the empirical mean of m_s from the gauged states, then searches for the
    value of 'g' where the gradient of the log partition function (theoretical m_s average) is
    closest to the empirical mean. Optionally, it can plot the comparison and print verbose output.

    Args:
        emp_states_dict_gauged (dict): Dictionary mapping state tuples to their empirical probabilities.
        plot_graph (bool, optional): If True, plots the theoretical m_s average, empirical mean,
            and the optimal g. Defaults to False.
        verbose (bool, optional): If True, prints the empirical mean, optimal g, and corresponding
            theoretical value. Defaults to False.

    Returns:
        float: The optimal value of g that best matches the empirical m_s mean.
    """

    ms_mean = calc_ms_mean(emp_states_dict_gauged)              # Empirical value to compare with theoretical ms_average values
    latent_dim = len(next(iter(emp_states_dict_gauged)))

    gs = np.linspace(-3, 3, 1000)                               # g domain to calculate -log(Z(g))

    y = []
    for g in gs:
        y.append(-np.log(calc_Z_theoretical(latent_dim, g)))    
    ms_average = np.gradient(y, gs)                             # Theoretical values for comparison with ms_mean. Numerical gradients of y over g (vector).

    nearest_position = np.argmin(np.abs(ms_average - ms_mean))
    nearest_g = gs[nearest_position]
    nearest_value = ms_average[nearest_position]

    # --------------------------------

    if IS_TEST_MODE:
        plot_graph = True
    if plot_graph:
        plt.plot(gs, ms_average, label="m_s average")
        plt.axhline(ms_mean, color="r", linestyle="--", label="ms_mean")
        plt.axvline(
            nearest_g, color="g", linestyle="--", label=f"optimal g = {nearest_g:.3f}"
        )
        plt.legend()
        plt.xlabel("g")
        plt.ylabel("Expected Value")
        plt.title("m_s average vs m_s mean")
        plt.show()

    if verbose:
        print(f"m_s mean: {ms_mean}")
        print(f"Optimal g: {nearest_g}, with expected value: {nearest_value}")

    return nearest_g




#–––––––––––––––––––––––––––––––––––––––– USED IN THE FUNCTIONS DEFINED ABOVE - 2 ––––––––––––––––––––––––––––––––––––––––––––––––––



def flip_gauge_bits(emp_states_dict, save_flip_gauge=False, model_kwargs=None):                       # USED IN compute_emp_states_dict_gauged
    """
    Flip specific bits in all states based on the activated bits in the most frequent state.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to their empirical probabilities.
                                From compute_emp_states_dict.

    Returns:
        dict: A new dictionary with the same probabilities but flipped states according to the rule.
    """

    most_frequent_state = max(emp_states_dict.items(), key=lambda x: x[1])[0]

    if save_flip_gauge and model_kwargs is not None:
        flip_save_dir = os.path.join(project_root, "gauges", "flip", model_kwargs['output_activation_encoder_path'], model_kwargs['output_activation_decoder_path'], str(model_kwargs['latent_dim']), model_kwargs['dataset'], model_kwargs['lm_lmb'], f"flipg_{model_kwargs['num_hidden_layers']}hl.txt")
        os.makedirs(os.path.dirname(flip_save_dir), exist_ok=True)
        train_num = model_kwargs.get('train_num', 'unknown')
        with open(flip_save_dir, 'a') as f:
            f.write(f"{train_num}\t{[int(x) for x in most_frequent_state]}\n")

    bits_to_flip = [i for i, bit in enumerate(most_frequent_state) if bit == 1]

    emp_gauged_states_dict = {}

    for state, prob in emp_states_dict.items():
        new_state = list(state)

        for bit_pos in bits_to_flip:
            new_state[bit_pos] = 1 - new_state[bit_pos]

        emp_gauged_states_dict[tuple(new_state)] = prob

    return emp_gauged_states_dict



# To be used to compute the best permutation of the states that minimizes the KL divergence with the HFM model.
# The same permutation should be valid for all g values, therefore it is not necessary to recalculate the gauge for different g values. See ```print_minimum_kl_in_g_range``` below.
def compute_perm_minimizing_hfm_kld_brute_force(
    emp_states_dict: dict,
    g = np.log(2),
    verbose = False,
):
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2). The best permutation should be independent from g.
        return_gauged_states_dict (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        best_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """


    state_len = len(list(emp_states_dict.keys())[0])

    permutations_list = list(permutations(range(state_len)))
    permutations_count = len(permutations_list)

    best_kl = float("inf")
    best_permutation = None

    i = 0
    for perm in permutations_list:
        i += 1

        permuted_states_dict = {tuple(k[i] for i in perm): v for k, v in emp_states_dict}

        current_kl = calc_hfm_kld(permuted_states_dict, g=g)
        if current_kl < best_kl:
            best_kl = current_kl
            best_permutation = perm

        if verbose:
            if i % 10000 == 0:
                print(
                    f"Processed {i}/{permutations_count} permutations, current minimum KL: {best_kl}, best permutation: {best_permutation}"
                )

    return best_permutation



def compute_perm_minimizing_hfm_kld_simul_anneal(
    emp_states_dict: dict,
    g = np.log(2),
    initial_temp = 10.0,
    cooling_rate = 0.95,
    n_iterations = 5000,
    hfm_distribution = 'pure',
    verbose = False
):
    """
    Uses simulated annealing to compute a permutation of state columns that minimizes
    the KL divergence between the empirical state distribution and the HFM model.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to probabilities, with bits already gauge flipped.
        g (float, optional): HFM model parameter. Defaults to np.log(2). The best permutation should be independent from g.
        initial_temp (float, optional): Initial temperature for simulated annealing. Defaults to 10.0.
        cooling_rate (float, optional): Cooling rate for temperature decay. Defaults to 0.95.
        n_iterations (int, optional): Number of iterations for the annealing process. Defaults to 5000.
        verbose (bool, optional): If True, prints progress every 100 iterations. Defaults to False.

    Returns:
        list: The permutation of state columns that yields the lowest KL divergence found.
    """

    state_len = len(list(emp_states_dict.keys())[0])

    current_perm = list(range(state_len))                   # Identity permutation
    current_states_dict = emp_states_dict
    if hfm_distribution == 'pure':
        current_kl = calc_hfm_kld(current_states_dict, g=g)
    elif hfm_distribution == 'marginalized':
        current_kl = calc_hfm_kld_with_marginalized_hfm(current_states_dict, g=g)

    best_perm = current_perm                                # No need to .copy() because the variable is not modified
    best_kl = current_kl

    # Metropolis algorithm
    temp = initial_temp
    for i in range(n_iterations):

        swap_indices = random.choice(list(combinations(range(state_len), 2)))   

        candidate_perm = current_perm.copy()                                    # .copy() because the variable is modified
        candidate_perm[swap_indices[0]], candidate_perm[swap_indices[1]] = (    # swapped using tuple unpacking
            current_perm[swap_indices[1]], current_perm[swap_indices[0]]
        )   

        permuted_states_dict = {tuple(k[i] for i in candidate_perm): v for k, v in emp_states_dict.items()}
        if hfm_distribution == 'pure':
            candidate_kl = calc_hfm_kld(permuted_states_dict, g=g)
        elif hfm_distribution == 'marginalized':
            candidate_kl = calc_hfm_kld_with_marginalized_hfm(permuted_states_dict, g=g)

        delta_kl = candidate_kl - current_kl
        if delta_kl < 0 or random.random() < math.exp(-delta_kl / temp):

            current_perm = candidate_perm
            current_kl = candidate_kl

            if current_kl < best_kl:
                best_perm = current_perm.copy()
                best_kl = current_kl

        temp *= cooling_rate

        # Periodic progress report
        if verbose and (i + 1) % 100 == 0:
            print(
                f"Iteration {i + 1}, Current KL: {current_kl:.6f}, Best KL: {best_kl:.6f}"
            )

    return best_perm



def calc_ms_mean(emp_states_dict_gauged):
    """
    Calculates the weighted mean of the m_s statistic over all states in the provided dictionary.

    For each state, computes m_s using calc_ms(state, False), then averages these values weighted
    by the empirical probabilities (frequencies) from emp_states_dict_gauged.

    Args:
        emp_states_dict_gauged (dict): Dictionary mapping state tuples to their empirical probabilities.

    Returns:
        float: Weighted mean of m_s across all states.
    """
    ms_values = [calc_ms(state, False) for state in emp_states_dict_gauged.keys()]
    weights = [emp_states_dict_gauged[state] for state in emp_states_dict_gauged.keys()]
    return np.average(ms_values, weights=weights)





