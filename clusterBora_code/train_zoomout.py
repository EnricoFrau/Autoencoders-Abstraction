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

    from AE.datasets import train_loader_2MNIST, val_loader_2MNIST, train_loader_MNIST, val_loader_MNIST, train_loader_EMNIST, val_loader_EMNIST, train_loader_FEMNIST, val_loader_FEMNIST
    from AE.models import AE_0
    from AE.train import layer_wise_pretrain_load_dict, train


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

    
    datasets = ["2MNIST", "MNIST", "EMNIST", "FEMNIST"]

    train_loaders = {
        "2MNIST": train_loader_2MNIST,
        "MNIST": train_loader_MNIST,
        "EMNIST": train_loader_EMNIST,
        "FEMNIST": train_loader_FEMNIST,
    }

    val_loaders = {
        "2MNIST": val_loader_2MNIST,
        "MNIST": val_loader_MNIST,
        "EMNIST": val_loader_EMNIST,
        "FEMNIST": val_loader_FEMNIST,
    }

    runs_dir = os.path.join(project_root, "runs", "zoomout")
    models_dir = os.path.join(project_root, "models", "zoomout")



    print("\n\n\n=================STARTING TRAINING QUANTIZED=================")
    for dataset in datasets:
        print(f"\n\n\n------------{dataset}------------\n")

        input_dim = 784
        learning_rate = 1e-3
        weight_decay = 1e-5
        decrease_rate = 0.625
        train_loader = train_loaders[dataset]
        val_loader = val_loaders[dataset]

        for train_num in range(6):
            for latent_dim in (10, 12, 14, 16):
                print(f"\n\n\n-----------------------Training models with {latent_dim} latent_dim----------------------\n\n\n")

                num_hidden_layers = 1

                print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")

                new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers=num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=True).to(device)
                log_dir = os.path.join(runs_dir, "quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}')
                writer = SummaryWriter(log_dir=log_dir)
                optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                train(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=40, device=device)
                os.makedirs(os.path.join(models_dir, "quantized", f'{latent_dim}ld', dataset), exist_ok=True)  
                save_dir = os.path.join(models_dir, "quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}.pth')
                torch.save(new_model.state_dict(), save_dir)

                ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=True).to(device)
                ex_model.load_state_dict(new_model.state_dict())

                for num_hidden_layers in range(2,8):
                    print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")
                    
                    new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=True).to(device)
                    layer_wise_pretrain_load_dict(ex_model, new_model)

                    log_dir = os.path.join(runs_dir, "quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}')
                    writer = SummaryWriter(log_dir=log_dir)
                    optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    train(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=40, device=device)
                    os.makedirs(os.path.join(models_dir, "quantized", f'{latent_dim}ld', dataset), exist_ok=True)
                    save_dir = os.path.join(models_dir, "quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}.pth')
                    torch.save(new_model.state_dict(), save_dir)

                    ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=True).to(device)
                    ex_model.load_state_dict(new_model.state_dict())

    print("\n\n\n=================TRAINING QUANTIZED ENDED=================\n\n\n")



    print("\n\n\n=================STARTING TRAINING NON QUANTIZED=================")
    for dataset in datasets:
        print(f"\n\n\n------------{dataset}------------\n")

        input_dim = 784
        learning_rate = 1e-3
        weight_decay = 1e-5
        decrease_rate = 0.625
        train_loader = train_loaders[dataset]
        val_loader = val_loaders[dataset]

        for train_num in range(6):
            for latent_dim in (10, 12, 14, 16):
                print(f"\n\n\n-----------------------Training models with {latent_dim} latent_dim----------------------\n\n\n")

                num_hidden_layers = 1

                print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")

                new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers=num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=False).to(device)
                log_dir = os.path.join(runs_dir, "non_quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}')
                writer = SummaryWriter(log_dir=log_dir)
                optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                train(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=40, device=device)
                os.makedirs(os.path.join(models_dir, "non_quantized", f'{latent_dim}ld', dataset), exist_ok=True)  
                save_dir = os.path.join(models_dir, "non_quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}.pth')
                torch.save(new_model.state_dict(), save_dir)

                ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=False).to(device)
                ex_model.load_state_dict(new_model.state_dict())

                for num_hidden_layers in range(2,8):
                    print(f"\n\n----------------- {num_hidden_layers} num_hidden_layers --------------\n\n")
                    
                    new_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=False).to(device)
                    layer_wise_pretrain_load_dict(ex_model, new_model)

                    log_dir = os.path.join(runs_dir, "non_quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}')
                    writer = SummaryWriter(log_dir=log_dir)
                    optimizer = optim.Adam(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    train(new_model, writer=writer, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=40, device=device)
                    os.makedirs(os.path.join(models_dir, "non_quantized", f'{latent_dim}ld', dataset), exist_ok=True)
                    save_dir = os.path.join(models_dir, "non_quantized", f'{latent_dim}ld', dataset, f'dr{decrease_rate}_{num_hidden_layers}hl_{train_num}.pth')
                    torch.save(new_model.state_dict(), save_dir)

                    ex_model = AE_0(input_dim=input_dim, latent_dim=latent_dim, decrease_rate=decrease_rate, hidden_layers = num_hidden_layers, output_activation_encoder=nn.Sigmoid, output_activation_decoder=nn.Sigmoid, quantize_latent=False).to(device)
                    ex_model.load_state_dict(new_model.state_dict())

    print("\n\n\n=================TRAINING NON QUANTIZED ENDED=================\n\n\n")




if __name__ == "__main__":
    main()
