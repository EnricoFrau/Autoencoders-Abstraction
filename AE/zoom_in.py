import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms



def select_data_by_feature_activation(
    model,
    dataset,
    permutation,
    feature_indices=(0,),
    gauge_flip=None,
    threshold=0.5,
    device="cpu",
):
    """
    Select data points that activate specific features after permutation.

    Args:
        model: trained autoencoder with quantized latent
        dataset: full dataset (e.g., EMNIST)
        feature_indices: list of feature indices to check (after permutation)
        permutation: permutation array from KLD minimization
        gauge_flip: optional gauge flip array
        threshold: activation threshold for binary features
        device: computation device
    
    Returns:
        indices of samples that activate the specified features
    """
    model.eval()
    selected_indices = []

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            data, _ = dataset[sample_idx]
            data = data.unsqueeze(0).to(device)

            latent = model.encode(data)  # expected shape: (1, latent_dim)

            # --- Apply gauge flip as XOR (bitwise) ---
            base_vec = latent[0].round().long()  # Assicurati che sia binario (0/1)
            gf = torch.as_tensor(gauge_flip, dtype=torch.long, device=latent.device)
            flipped = (base_vec ^ gf)  # XOR bitwise

            # --- Apply permutation ---
            perm = torch.as_tensor(permutation, dtype=torch.long, device=latent.device)
            latent_permuted = flipped[perm].float().unsqueeze(0)


            # --- Robust feature selection handling ---
            feats = torch.as_tensor(feature_indices, dtype=torch.long, device=latent.device).view(-1)

            is_active = (latent_permuted[0, feats] >= threshold).all().item()
            if is_active:
                selected_indices.append(sample_idx)

    return selected_indices



class BlurredSubset(Dataset):
    def __init__(self, base_dataset, indices, kernel_size, sigma):
        if indices is None:
            self.subset = base_dataset
        else:
            self.subset = Subset(base_dataset, indices)
        if kernel_size == 0 and sigma == 0:
            self.blur = None
        else:
            self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.blur is not None:
            img = self.blur(img)
        return img, label
