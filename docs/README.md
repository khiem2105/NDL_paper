# Documentation

This directory contains detailed documentation for the Network Dictionary Learning (NDL) codebase.

## Available Documentation

### Method Explanations

- **[get_patches_method_explanation.md](get_patches_method_explanation.md)** - Comprehensive explanation of the `get_patches` method in the `Network_Reconstructor` class, including:
  - Detailed parameter descriptions
  - Step-by-step algorithm walkthrough
  - Return value explanations
  - Usage examples
  - Performance considerations
  - Relationship to other methods in the framework

## Quick Reference

### Core Classes

- **`Network_Reconstructor`** (in `utils/ndl.py`) - Main class for Network Dictionary Learning and Reconstruction

### Key Methods

1. **`get_patches(B, emb, skip_folded_hom, sample_size, omit_folded_edges, sampling_alg)`**
   - Samples network patches using MCMC methods
   - Used for dictionary learning and network reconstruction
   
2. **`train_dict(...)`**
   - Learns a dictionary of network motifs from sampled patches
   
3. **`reconstruct_network(...)`**
   - Reconstructs or denoises a network using the learned dictionary

## Additional Resources

For the main project README, see [../README.md](../README.md)

For the research paper, see:
- Lyu, H., Kureh, Y., Vendrow, J., & Porter, M. A. (2021). "Learning low-rank latent mesoscale structures in networks." [arXiv:2102.06984](https://arxiv.org/abs/2102.06984)
