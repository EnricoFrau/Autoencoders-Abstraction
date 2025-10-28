This repository contains the code used to develop the analysis (and not only) of the Autoencoder Section in the paper [Absolute Abstraction: a renormalization group approach](https://arxiv.org/html/2407.01656v4).

Here can be found an overview of the repository content:
- Research codebase for studying abstraction in autoencoders by training and comparing models across depth and latent sizes on MNIST-family datasets (2MNIST, 2MNISTonly, MNIST, EMNIST, FEMNIST).
- Core PyTorch models: AE.models.AE_0 and AE.models.ProgressiveAE; training loop in AE.train.train.
- Analysis utilities compute HFM KL divergence, optimal binarization thresholds, gauge permutations, and activation frequencies, e.g. AE.depth_utils.calc_hfm_kld_with_optimal_g and AE.depth_utils.compute_emp_states_dict_gauged.
- Visualization helpers plot KL vs depth and latent features, e.g. AE.plotter_functions.plot_KLs_vs_hidden_layers and AE.plotter_functions.visualize_bottleneck_neurons.
- Overlap tools measure decoded-feature overlaps and distances across datasets, e.g. AE.overlaps.compute_overlap_matrix and AE.overlaps.plot_overlap_heatmap.
- Reproducible pipelines live in notebooks like depth_analysis.ipynb and experiments.ipynb; results are saved under savings and preliminary outputs under preliminary data.
