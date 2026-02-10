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


    from AE.utils import load_model
    from AE.utils import calc_MSE_loss
    from AE.datasets import train_loader_MNIST, train_loader_EMNIST, train_loader_2MNIST, train_loader_FEMNIST
    from AE.datasets import val_loader_2MNIST, val_loader_MNIST, val_loader_EMNIST, val_loader_FEMNIST

    train_loaders = {
        '2MNIST': train_loader_2MNIST,
        'MNIST': train_loader_MNIST,
        'EMNIST': train_loader_EMNIST,
        'FEMNIST': train_loader_FEMNIST
    }

    val_loaders = {
        '2MNIST': val_loader_2MNIST,
        'MNIST': val_loader_MNIST,
        'EMNIST': val_loader_EMNIST,
        'FEMNIST': val_loader_FEMNIST
    }


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

    if IS_CLUSTER_ENV and "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    print("Num threads:", torch.get_num_threads())

    SEED = 57
    torch.manual_seed(SEED)



# ------------------------------------------------------------------------

    latent_dims = [10,12,14,16]
    train_nums = range(6)
    datasets = ("2MNIST", "MNIST", "EMNIST", "FEMNIST")
    num_hidden_layers_range = range(1,8)

    for latent_dim in latent_dims:
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
                'dataset': dataset,
                'train_num': 0,
                'quantize_latent': True,
                'quantize_latent_path': 'quantized',
                'num_hidden_layers': 3,
                #'num_latent_samples': None
            }

            if i == 0:
                rep_dataset_train_loss_dict = {}
                rep_dataset_val_loss_dict = {}


            for train_num in train_nums:
                model_kwargs['train_num'] = train_num
                if train_num not in rep_dataset_train_loss_dict:
                    rep_dataset_train_loss_dict[train_num] = {"2MNIST": [], 'MNIST': [], 'EMNIST': [], 'FEMNIST': []}
                if train_num not in rep_dataset_val_loss_dict:
                    rep_dataset_val_loss_dict[train_num] = {"2MNIST": [], 'MNIST': [], 'EMNIST': [], 'FEMNIST': []}

                for num_hidden_layers in num_hidden_layers_range:
                    model_kwargs['num_hidden_layers'] = num_hidden_layers

                    model = load_model(model_kwargs, device)

                    rep_dataset_train_loss_dict[train_num][dataset].append(calc_MSE_loss(model, train_loaders[dataset], device=device))
                    rep_dataset_val_loss_dict[train_num][dataset].append(calc_MSE_loss(model, val_loaders[dataset], device=device))


        save_dir = os.path.join(project_root, 'savings', 'zoomout', 'losses', 'quantized', f'{latent_dim}ld')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'rep_dataset_train_loss_dict.pkl'), 'wb') as f:
            pickle.dump(rep_dataset_train_loss_dict, f)
        with open(os.path.join(save_dir, 'rep_dataset_val_loss_dict.pkl'), 'wb') as f:
            pickle.dump(rep_dataset_val_loss_dict, f)



if __name__ == "__main__":
    main()
