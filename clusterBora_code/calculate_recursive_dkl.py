def main():
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from torch import nn
    import pickle

    IS_TEST_MODE = True
    IS_CLUSTER_ENV = True

    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    from AE.datasets import train_loader_MNIST, val_loader_MNIST
    from AE.depth_utils import compute_dataset_klds_gs_dict_from_sampled_binarized_vectors_


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
            if IS_TEST_MODE:
                print("Set float32 matmul precision to high")
        except Exception:
            pass
        
    print("num_workers =", min(8, os.cpu_count() or 1))

    if IS_CLUSTER_ENV:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    print("Num threads:", torch.get_num_threads())

    SEED = 57
    torch.manual_seed(SEED)


    datasets = ["MNIST"]

    train_loaders = {
        "MNIST": train_loader_MNIST,
    }

    val_loaders = {
        "MNIST": val_loader_MNIST,
    }





    latent_dim = 10

    for i, lm_lmb in enumerate(('0.1',)):

        dataset = "MNIST"
        model_kwargs = {
            'input_dim': 28*28,
            'latent_dim': latent_dim,
            'decrease_rate': 0.6,
            'device': device,
            'output_activation_encoder': nn.Sigmoid,
            'output_activation_decoder': nn.Sigmoid,
            'output_activation_encoder_path': 'sigmoid output encoder',
            'output_activation_decoder_path': 'sigmoid output decoder',
            'train_type': 'simultaneous train',
            'dataset': dataset,
            'train_num': 0,
            'lm_lmb': str(lm_lmb),
            'recursive_last_layer': True,
        }

        if i == 0:
            lm_lmb_klds_dict_original = {}
            lm_lmb_gs_dict_original = {}

        num_hidden_layers_range = range(1,8)
        for train_num in range(2):
            model_kwargs['train_num'] = train_num

            if train_num not in lm_lmb_klds_dict_original:
                lm_lmb_klds_dict_original[train_num] = {'0.1': [], "0.3": [], '0.5': [], '0.7': [], '0.9': []}
            if train_num not in lm_lmb_gs_dict_original:
                lm_lmb_gs_dict_original[train_num] = {'0.1': [], "0.3": [], '0.5': [], '0.7': [], '0.9': []}

            lm_lmb_klds_dict_original[train_num], lm_lmb_gs_dict_original[train_num] = compute_dataset_klds_gs_dict_from_sampled_binarized_vectors_(
                dataset=lm_lmb,
                data_loader=train_loaders[dataset],
                model_kwargs=model_kwargs,
                device=device,
                num_hidden_layers_range=num_hidden_layers_range,
                dataset_klds_dict=lm_lmb_klds_dict_original[train_num],
                dataset_gs_dict=lm_lmb_gs_dict_original[train_num],
                save_permutations=True,
            )

    os.makedirs(os.path.join(project_root, 'savings', 'recursive', f'{latent_dim}ld'), exist_ok=True)
    with open(os.path.join(project_root, 'savings', 'recursive', f'{latent_dim}ld', 'dataset_klds_dict_sigmoid_output.pkl'), 'wb') as f:
        pickle.dump(lm_lmb_klds_dict_original, f)
    with open(os.path.join(project_root, 'savings', 'recursive', f'{latent_dim}ld', 'dataset_gs_dict_sigmoid_output.pkl'), 'wb') as f:
        pickle.dump(lm_lmb_gs_dict_original, f)


if __name__ == "__main__":
    main()
