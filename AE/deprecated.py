

def compute_dataset_klds_gs_dict_with_optimal_threshold_(dataset, data_loader, device, model_kwargs, model_path_kwargs, binarize_threshold_range, num_hidden_layers_range, dataset_klds_dict = None, dataset_gs_dict = None, verbose=True):  # DEPRECATED
    """
    Computes and stores the KLDs and gs values for a given dataset by searching for the optimal binarization threshold
    that minimizes a custom energy function.

    Args:
        dataset (str): Name of the dataset (e.g., 'MNIST', 'EMNIST', '2MNIST').
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        model_kwargs (dict): Keyword arguments for constructing the model (passed to AE_0).
        model_path_kwargs (dict): Keyword arguments for constructing the model path.
        binarize_threshold_range (iterable): Range of binarization thresholds to search over.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        dataset_klds_dict (dict, optional): Dictionary to store KLDs for each dataset. If None, a new one is created.
        dataset_gs_dict (dict, optional): Dictionary to store gs values for each dataset. If None, a new one is created.
        verbose (bool, optional): If True, prints the best threshold and energy found.

    Returns:
        tuple: (dataset_klds_dict, dataset_gs_dict) with the results for the given dataset and optimal threshold.
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

    dataset_klds_dict[dataset] = []
    dataset_gs_dict[dataset] = []

    memoize_klds_dict = {}
    memoize_gs_dict = {}

    klds_sign_changes_lst, gs_distances_lst = compute_energy_addends_lst(
        binarize_threshold_range=binarize_threshold_range, 
        dataset=dataset,
        device=device, 
        data_loader=data_loader, 
        model_kwargs=model_kwargs, 
        model_path_kwargs=model_path_kwargs, 
        num_hidden_layers_range=num_hidden_layers_range, 
        memoize_klds_dict=memoize_klds_dict, 
        memoize_gs_dict=memoize_gs_dict
    )

    best_binarize_thresold, best_energy = optimize_binarize_threshold_through_energy(
        klds_sign_changes_lst, 
        gs_distances_lst, 
        binarize_threshold_range
    )

    dataset_klds_dict[dataset] = memoize_klds_dict[str(best_binarize_thresold)]
    dataset_gs_dict[dataset] = memoize_gs_dict[str(best_binarize_thresold)]

    if verbose:
        print(f"best_binarize_thresold: ", best_binarize_thresold)
        print(f"best_energy: ", best_energy)

    return dataset_klds_dict, dataset_gs_dict



# –––––––––––––––––––––––––– NEEDED FOR compute_dataset_klds_gs_dict_with_optimal_threshold_ ---- DEPRECATED ––––––––––––––––––––––––––––––––––––––



def optimize_binarize_threshold_through_energy(klds_sign_changes_lst, gs_distances_lst, binarize_threshold_range):
    """
    Finds the optimal binarization threshold that minimizes a custom energy function.

    The energy function is defined as the sum of the normalized number of KLD sign changes
    and the normalized sum of distances of gs values from log(2), for each threshold.

    Args:
        klds_sign_changes_lst (list or np.ndarray): List of normalized KLD sign change counts for each threshold.
        gs_distances_lst (list or np.ndarray): List of normalized gs distances from log(2) for each threshold.
        binarize_threshold_range (iterable): Range of binarization thresholds considered.

    Returns:
        tuple: (best_binarize_threshold, best_energy)
            - best_binarize_threshold (float): The threshold value that minimizes the energy.
            - best_energy (float): The minimum energy value found.
    """
    best_energy = float('inf')
    best_binarize_thresold = None

    for i, binarize_threshold in enumerate(binarize_threshold_range):
        energy = (
            3 * klds_sign_changes_lst[i] 
            + 1 * gs_distances_lst[i]
        )
        if energy < best_energy:
            best_energy = energy
            best_binarize_thresold = binarize_threshold

    return best_binarize_thresold, best_energy



def compute_energy_addends_lst(binarize_threshold_range, dataset, data_loader, device, model_kwargs, model_path_kwargs, num_hidden_layers_range, memoize_klds_dict, memoize_gs_dict):
    """
    Computes the normalized energy addends for a range of binarization thresholds.

    For each threshold in `binarize_threshold_range`, this function:
      - Computes the list of KLDs and gs values across different hidden layer counts.
      - Calculates the number of sign changes in the KLDs list.
      - Calculates the sum of absolute differences between each gs value and log(2).
      - Normalizes both lists by their respective maxima.

    Args:
        binarize_threshold_range (iterable): Range of binarization thresholds to evaluate.
        dataset (str): Name of the dataset.
        data_loader (DataLoader): DataLoader for the dataset.
        model_kwargs (dict): Arguments for model construction.
        model_path_kwargs (dict): Arguments for model path construction.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        memoize_klds_dict (dict): Dictionary to store computed KLDs for each threshold.
        memoize_gs_dict (dict): Dictionary to store computed gs values for each threshold.

    Returns:
        tuple:
            - klds_sign_changes_lst (np.ndarray): Normalized list of KLD sign change counts for each threshold.
            - gs_distances_lst (np.ndarray): Normalized list of gs distances from log(2) for each threshold.
    """
    klds_sign_changes_lst = []
    gs_distances_lst = []

    for binarize_threshold in binarize_threshold_range:

        klds_lst, gs_lst = compute_klds_gs_lst_with_fixed_threshold(
            data_loader=data_loader, 
            device=device, 
            model_kwargs=model_kwargs, 
            model_path_kwargs=model_path_kwargs, 
            binarize_threshold=binarize_threshold,
            num_hidden_layers_range=num_hidden_layers_range, 
            memoize_klds_dict=memoize_klds_dict, 
            memoize_gs_dict=memoize_gs_dict)

        klds_sign_changes_lst.append( count_sign_changes(klds_lst) )
        gs_distances_lst.append( sum(abs(a - np.log(2)) for a in gs_lst) )

    klds_sign_changes_lst = np.array(klds_sign_changes_lst) / max(klds_sign_changes_lst)
    gs_distances_lst = np.array(gs_distances_lst) / max(gs_distances_lst)

    return klds_sign_changes_lst, gs_distances_lst




def compute_klds_gs_lst_with_fixed_threshold(data_loader, model_kwargs, device, model_path_kwargs, binarize_threshold, num_hidden_layers_range, memoize_klds_dict, memoize_gs_dict):
    """
    Computes lists of KLD and gs values for a fixed binarization threshold across a range of hidden layer counts.

    For each number of hidden layers in `num_hidden_layers_range`, this function:
      - Loads the corresponding model using the provided `model_kwargs` and `model_path_kwargs`.
      - Computes the Kullback-Leibler divergence (KLD) and gs value using `calc_hfm_kld_with_optimal_g`.
      - Appends the results to lists.

    Optionally memoizes the results for the given threshold in the provided dictionaries.

    Args:
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        model_kwargs (dict): Arguments for constructing the model (passed to AE_0).
        model_path_kwargs (dict): Arguments for constructing the model path.
        binarize_threshold (float): The binarization threshold to use.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        memoize_klds_dict (dict): Dictionary to store computed KLDs for each threshold.
        memoize_gs_dict (dict): Dictionary to store computed gs values for each threshold.

    Returns:
        tuple: (klds_lst, gs_lst)
            - klds_lst (list): List of KLD values for each hidden layer count.
            - gs_lst (list): List of gs values for each hidden layer count.
    """

    klds_lst = []
    gs_lst = []

    for num_hidden_layers in num_hidden_layers_range:

        my_model = AE_0(
            **model_kwargs,
            hidden_layers=num_hidden_layers
        ).to(device)
        model_path = f"../models/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['initialization']}/{model_path_kwargs['train_type']}/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/lr{model_path_kwargs['learning_rate']}_dr{model_path_kwargs['decrease_rate']}_bias{model_path_kwargs['bias']}_{num_hidden_layers}hl_{model_path_kwargs['train_num']}.pth"
        my_model.load_state_dict(torch.load(model_path, map_location=device))

        current_kld, current_g = calc_hfm_kld_with_optimal_g(my_model, data_loader, return_g=True, binarize_threshold=binarize_threshold)
        
        gs_lst.append(current_g)
        klds_lst.append(current_kld)

    if memoize_klds_dict is not None:
        memoize_klds_dict[str(binarize_threshold)] = klds_lst
    if memoize_gs_dict is not None:
        memoize_gs_dict[str(binarize_threshold)] = gs_lst

    return klds_lst, gs_lst



def count_sign_changes(values):
    """
    Counts the number of sign changes in the first differences of a sequence.
    Args:
        values (list or np.ndarray): Input sequence of numbers.
    Returns:
        int: The computed 'sign_changes' (number of sign changes).
    """
    diff_0 = 0
    sign_changes = 0
    for a, b in zip(values[:-1], values[1:]):
        diff_1 = b - a
        if diff_0 == 0:
            diff_0 = diff_1
            continue
        if np.sign(diff_1) != np.sign(diff_0):
            sign_changes += 1*abs(diff_1)
        diff_0 = diff_1
    return sign_changes





#=================================================================================================







def calc_optimal_g_with_marginalized_hfm(
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

    gs = np.linspace(np.log(2)+1e-4, 1.8, 1000)                               # g domain to calculate -log(Z(g))

    y = []
    for g in gs:
        y.append(calc_hfm_marginalized_log(emp_states_dict_gauged, g))    
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


def calc_hfm_marginalized_log(emp_states_dict_gauged, g):

    n = len(next(iter(emp_states_dict_gauged)))
    states = list(emp_states_dict_gauged.keys())
    x = 0
    for state in states:
        m_s = calc_ms(state)
        denom = np.exp(g) - 1
        x += np.log(1 - 1/denom + (1/denom) * np.exp(-g *(n - m_s)))
    return x / len(states)




def complementary_average_ms(emp_states_dict_gauged, g):
    
    n = len(next(iter(emp_states_dict_gauged)))
    states = list(emp_states_dict_gauged.keys())

    x = 0
    for state in states:
        m_s = calc_ms(states)
        x += (n - m_s - 1) * np.exp(-g * (n - m_s))

    return x / len(states)
