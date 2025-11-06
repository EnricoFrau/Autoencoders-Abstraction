This repository contains the code used to develop the analysis (and not only) of the Autoencoder Section in the paper [Absolute Abstraction: a renormalization group approach](https://arxiv.org/html/2407.01656v4).

Here can be found an overview of the repository content:

- **datasets.py**: Utilities for loading and preprocessing MNIST-family datasets for training and evaluation.
- **depth_utils.py**: Functions for analyzing autoencoder depth, including KL divergence computation, binarization thresholds, and activation statistics.
- **models.py**: Implementation of core PyTorch autoencoder models, including baseline (`AE_0`) and progressive architectures.
- **overlaps.py**: Tools for measuring and visualizing feature overlaps and distances between decoded representations.
- **plotter_functions.py**: Visualization utilities for plotting KL divergences, latent features, and bottleneck neuron activations.
- **train.py**: Training loop and routines for fitting autoencoder models to datasets.
- **utils.py**: General helper functions used throughout the module.
