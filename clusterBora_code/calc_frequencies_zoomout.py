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

    from AE.overlaps import get_datapoints_labels_arrays, compute_rep_hl_datapoints_labels_freq



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



    latent_dims = [10,]
    datasets = ["2MNIST"]
    train_nums = range(6)



    for dataset in datasets:

        datapoints_array, labels_array = get_datapoints_labels_arrays(dataset, train=True)


        for latent_dim in latent_dims:

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
                #'num_latent_samples': None
            }

            save_dir = os.path.join(project_root, "savings/zoomout/features_freq/quantized", f"{latent_dim}ld", dataset)
            os.makedirs(save_dir, exist_ok=True)

            rep_hl_datapoints_freq, rep_hl_labels_freq, rep_hl_distances  = compute_rep_hl_datapoints_labels_freq(
                model_kwargs=model_kwargs,
                datapoints_array=datapoints_array,
                labels_array=labels_array,
                repetitions_range=train_nums,
                num_hidden_layers_range=range(1, 8),
                return_distances=True,
                save_dir=save_dir
            )

            with open(f"{save_dir}/rep_hl_datapoints_freq.pkl", "wb") as f:
                pickle.dump(rep_hl_datapoints_freq, f)
            with open(f"{save_dir}/rep_hl_labels_freq.pkl", "wb") as f:
                pickle.dump(rep_hl_labels_freq, f)
            with open(f"{save_dir}/rep_hl_distances.pkl", "wb") as f:
                pickle.dump(rep_hl_distances, f)







if __name__ == "__main__":
    main()
