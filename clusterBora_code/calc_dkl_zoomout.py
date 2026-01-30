def main():
    import torch
    from torch import nn
    import pickle

    IS_TEST_MODE = True
    IS_CLUSTER_ENV = True

    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    from AE.datasets import train_loader_2MNIST, train_loader_MNIST, train_loader_EMNIST, train_loader_FEMNIST
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

    
    datasets = ["2MNIST","MNIST"]

    train_loaders = {
        "2MNIST": train_loader_2MNIST,
        "MNIST": train_loader_MNIST,
        "EMNIST": train_loader_EMNIST,
        "FEMNIST": train_loader_FEMNIST,
    }

    for latent_dim in (10,):
        for i, dataset in enumerate(datasets):

            model_kwargs = {
                'input_dim': 28*28,
                'latent_dim': latent_dim,
                'decrease_rate': 0.625,
                'device': device,
                'output_activation_encoder': nn.Sigmoid,
                'output_activation_decoder': nn.Sigmoid,
                'output_activation_encoder_path': 'sigmoid output encoder',
                'output_activation_decoder_path': 'sigmoid output decoder',
                'train_type': 'simultaneous train',
                'dataset': dataset,
                'train_num': 0,
                'quantize_latent': True,
                'quantize_latent_path': 'quantized'
            }

            if i == 0:
                dataset_klds_dict = {}
                dataset_gs_dict = {}

            num_hidden_layers_range = range(1,8)
            for train_num in range(3):
                model_kwargs['train_num'] = train_num

                if train_num not in dataset_klds_dict:
                    dataset_klds_dict[train_num] = {'2MNIST': [], 'MNIST': [], 'EMNIST': [], 'FEMNIST': []}
                if train_num not in dataset_gs_dict:
                    dataset_gs_dict[train_num] = {'2MNIST': [], 'MNIST': [], 'EMNIST': [], 'FEMNIST': []}

                gauges_dir = os.path.join('zoomout', 'quantized', f'{latent_dim}ld', f'{dataset}')

                dataset_klds_dict[train_num], dataset_gs_dict[train_num] = compute_dataset_klds_gs_dict_from_sampled_binarized_vectors_(
                    dataset=dataset,
                    data_loader=train_loaders[dataset],
                    model_kwargs=model_kwargs,
                    num_hidden_layers_range=num_hidden_layers_range,
                    dataset_klds_dict=dataset_klds_dict[train_num],
                    dataset_gs_dict=dataset_gs_dict[train_num],
                    save_gauges_dir=gauges_dir,
                )

        os.makedirs(os.path.join(project_root, 'savings', 'zoomout', 'klds_gs', 'quantized', f'{latent_dim}ld'), exist_ok=True)
        with open(os.path.join(project_root, 'savings', 'zoomout', 'klds_gs', 'quantized', f'{latent_dim}ld', 'dataset_klds_dict.pkl'), 'wb') as f:
            pickle.dump(dataset_klds_dict, f)
        with open(os.path.join(project_root, 'savings', 'zoomout','klds_gs', 'quantized', f'{latent_dim}ld', 'dataset_gs_dict.pkl'), 'wb') as f:
            pickle.dump(dataset_gs_dict, f)


if __name__ == "__main__":
    main()
