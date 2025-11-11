def main():
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from torch import nn

    IS_TEST_MODE = True
    IS_CLUSTER_ENV = True

    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    from AE.datasets import train_loader_MNIST, val_loader_MNIST
    from AE.models import AE_0
    from AE.train import train_recursiveAE, layer_wise_pretrain_load_dict


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



    runs_dir = os.path.join(project_root, "runs", "recursive")
    models_dir = os.path.join(project_root, "models", "recursive")


    print("\n\n\n=================STARTING TRAINING=================")
    for dataset in ("MNIST",):
        print(f"\n\n\n------------{dataset}------------\n\n\n")

        input_dim = 784
        learning_rate = 1e-3
        learning_rate_str = "1e-3"
        decrease_rate_str = "0.6"
        weight_decay = 1e-5
        decrease_rate = 0.6
        train_loader = train_loaders[dataset]
        val_loader = val_loaders[dataset]
        train_num = 0


        for latent_dim in (10,):
            print(f"-----------------------{latent_dim} LATENT_DIM----------------------")

            num_hidden_layers = 1

            print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")
            
            new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, recursive_last_layer=True, he_init=False).to(device)
            
            writer = SummaryWriter(log_dir=os.path.join(runs_dir, f'{latent_dim}ld', dataset, f'dr{decrease_rate_str}_lr{learning_rate_str}_lwpretrain_{num_hidden_layers}hl_{train_num}'))
            optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            train_recursiveAE(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, device=device, optimizer=optimizer, epochs=10, detach_encoded_target=False)
            save_dir = os.path.join(models_dir, f'{latent_dim}ld', dataset, f'dr{decrease_rate_str}_lr{learning_rate_str}_lwpretrain_{num_hidden_layers}hl_{train_num}.pth')
            torch.save(new_model.state_dict(), save_dir)

            ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, recursive_last_layer=True).to(device)
            ex_model.load_state_dict(new_model.state_dict())

            for num_hidden_layers in range(2,8):
                print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")
                
                new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, recursive_last_layer=True, he_init=False).to(device)
                layer_wise_pretrain_load_dict(ex_model, new_model)

                writer = SummaryWriter(log_dir=os.path.join(runs_dir, f'{latent_dim}ld', dataset, f'dr{decrease_rate_str}_lr{learning_rate_str}_lwpretrain_{num_hidden_layers}hl_{train_num}'))
                optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                train_recursiveAE(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, device=device,optimizer=optimizer, epochs=10, detach_encoded_target=False)
                save_dir = os.path.join(models_dir, f'{latent_dim}ld', dataset, f'dr{decrease_rate_str}_lr{learning_rate_str}_lwpretrain_{num_hidden_layers}hl_{train_num}.pth')
                torch.save(new_model.state_dict(), save_dir)

                ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, recursive_last_layer=True).to(device)
                ex_model.load_state_dict(new_model.state_dict())
        
    print("\n\n\n=================TRAINING ENDED=================\n\n\n")


if __name__ == "__main__":
    main()
