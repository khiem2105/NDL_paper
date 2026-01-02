# Detailed Explanation of the `get_patches` Method in Network_Reconstructor Class

## Overview

The `get_patches` method is a core component of the Network Dictionary Learning (NDL) framework. It samples network patches (subgraphs) from a network using Markov Chain Monte Carlo (MCMC) methods. These patches are used for dictionary learning and network reconstruction tasks.

## Method Signature

```python
def get_patches(self, B, emb,
                skip_folded_hom=False,
                sample_size=1,
                omit_folded_edges=False,
                sampling_alg='pivot'):
```

## Parameters

### Required Parameters:
- **`B`** (numpy.ndarray): Adjacency matrix of the motif F (a template graph structure) to be embedded into the network. Shape: (k, k) where k is the number of nodes in the motif.
- **`emb`** (array-like): Current embedding of the motif into the network. This represents a mapping from motif nodes to network nodes (F → G).

### Optional Parameters:
- **`skip_folded_hom`** (bool, default=False): If True, skips adding sampled patches where nodes are not distinct (i.e., when len(set(emb)) < k). This ensures only injective homomorphisms are collected.
- **`sample_size`** (int, default=1): Number of patches to sample from the network.
- **`omit_folded_edges`** (bool, default=False): If True, tracks which edges appear due to "folding" (when distinct motif nodes map to the same network node) and returns this information.
- **`sampling_alg`** (str, default='pivot'): The MCMC algorithm to use for sampling. Options:
  - `'pivot'`: Pivot chain sampling
  - `'glauber'`: Glauber dynamics
  - `'idla'`: Internal Diffusion Limited Aggregation
  - `'pivot_inj'`: Injective pivot sampling

## Return Values

The method returns different values depending on the `omit_folded_edges` parameter:

- **If `omit_folded_edges=False`**: Returns `(X, emb)` where:
  - `X`: Matrix of sampled patches, shape (k², sample_size) for regular networks or (k², color_dim, sample_size) for tensor networks
  - `emb`: Updated embedding after sampling

- **If `omit_folded_edges=True`**: Returns `(X, emb, nofolding_indicator)` where:
  - `X`: Matrix of sampled patches (same as above)
  - `emb`: Updated embedding after sampling
  - `nofolding_indicator`: Binary matrix indicating which edges are not due to folding

## How the Method Works

### 1. **Initialization** (Lines 412-419)

```python
k = B.shape[0]  # Size of the motif (number of nodes)
X = np.zeros((k ** 2, 1))  # Initialize patch storage
if self.if_tensor_ntwk:
    X = np.zeros((k ** 2, self.G.color_dim, 1))

num_hom_sampled = 0  # Counter for successfully sampled homomorphisms
X = []  # Will store list of patches
count = 0  # Total iteration counter
```

The method:
- Determines the motif size `k` from the adjacency matrix `B`
- Initializes storage for patches `X` (accounting for tensor networks with colored edges if needed)
- Sets up counters to track sampling progress

### 2. **Sampling Loop** (Lines 421-437)

```python
while (num_hom_sampled < sample_size) and (count < 10000 * sample_size):
    # Sample a mesoscale patch
    meso_patch = self.update_hom_get_meso_patch(B, emb,
                                                iterations=1,
                                                sampling_alg=sampling_alg,
                                                omit_folded_edges=omit_folded_edges)
    Y = meso_patch[0]  # The adjacency matrix of the sampled patch
    emb = meso_patch[1]  # Updated embedding
    
    # Filter based on skip_folded_hom condition
    if not (skip_folded_hom and len(set(emb))<k):
        # Reshape patch for storage
        if not self.if_tensor_ntwk:
            Y = Y.reshape(k ** 2, -1)
        else:
            Y = Y.reshape(k ** 2, self.G.color_dim, -1)
        X.append(Y)
        num_hom_sampled += 1
    count += 1
```

**Key steps in each iteration:**

a. **Call `update_hom_get_meso_patch`**: This helper method does the actual work:
   - Updates the embedding using the specified MCMC algorithm (Glauber/Pivot/IDLA)
   - Extracts the induced subgraph from the network based on the new embedding
   - Returns the adjacency matrix of this subgraph as a "patch"

b. **Extract results**:
   - `meso_patch[0]`: The k×k adjacency matrix representing edge weights/presence between embedded nodes
   - `meso_patch[1]`: The updated embedding (mapping from motif to network nodes)
   - `meso_patch[2]` (if `omit_folded_edges=True`): Indicator matrix for non-folded edges

c. **Filter based on node distinctness**:
   - If `skip_folded_hom=True`, only accept patches where all k nodes in the embedding are distinct
   - This ensures the embedding is injective (one-to-one mapping)

d. **Reshape and store**:
   - Flatten the k×k adjacency matrix into a k² dimensional vector
   - For tensor networks, maintain the color dimension
   - Append to the list of patches

e. **Safety counter**:
   - The loop has a maximum iteration limit of `10000 * sample_size` to prevent infinite loops
   - This is important when `skip_folded_hom=True` makes sampling difficult

### 3. **Post-processing** (Lines 438-446)

```python
if len(X) > 0:
    X = np.asarray(X)[..., 0].T  # Convert list to array and transpose
else:
    X = None  # No patches were successfully sampled

if not omit_folded_edges:
    return X, emb
else:
    return X, emb, meso_patch[2]
```

**Final processing:**
- Converts the list of patches into a numpy array
- Transposes to get shape (k², sample_size) where each column is one patch
- Returns None if no patches were successfully sampled
- Includes the folding indicator matrix if requested

## Role in the NDL Framework

The `get_patches` method serves several critical purposes:

### 1. **Dictionary Learning** (`train_dict` method)
- Samples many patches from the network
- These patches form the training data for learning a dictionary of common subgraph patterns
- The dictionary atoms represent recurring local structures in the network

### 2. **Network Reconstruction** (`reconstruct_network` methods)
- Iteratively samples patches from the network
- Each patch is compared to the learned dictionary
- Reconstructed patches are aggregated to rebuild the network structure
- Used for denoising and link prediction tasks

### 3. **Network Analysis**
- Patches capture local mesoscale structures
- The distribution of patches reveals network motifs and patterns
- Can be used to compare different networks

## Relationship to Other Methods

### Helper Method: `update_hom_get_meso_patch`
This method is called internally and performs:
1. **MCMC Update**: Updates the embedding using one of the sampling algorithms
2. **Induced Subgraph Extraction**: Gets the actual edges in the network between embedded nodes
3. **Averaging**: Can average over multiple iterations (though `get_patches` uses iterations=1)

### Sampling Algorithms Used:
1. **Glauber Dynamics** (`glauber_gen_update`):
   - Randomly selects a node in the motif
   - Resamples its image in the network
   - Maintains the homomorphism property (respects the motif structure)

2. **Pivot Chain** (`Pivot_update`):
   - Updates a "pivot" node using random walk
   - Resamples the entire path/motif from the new pivot
   - Generally mixes faster than Glauber

3. **IDLA** (Internal Diffusion Limited Aggregation):
   - Samples k distinct nodes from the network
   - Creates an injective embedding
   - Different mixing properties

## Example Usage

```python
# Initialize Network Reconstructor
G = NNetwork()  # Your network
reconstructor = Network_Reconstructor(G, n_components=100, k2=4)

# Set up motif (e.g., a path of length 4)
B = reconstructor.path_adj(k1=0, k2=4)

# Initialize embedding
x0 = np.random.choice(list(G.vertices()))
emb = reconstructor.tree_sample(B, x0)

# Sample 100 patches using pivot algorithm
X, emb = reconstructor.get_patches(
    B=B, 
    emb=emb,
    sample_size=100,
    sampling_alg='pivot',
    skip_folded_hom=True  # Only accept injective embeddings
)

# X now contains 100 vectorized patches of the network
# Shape: (25, 100) for a 5-node motif (5² = 25)
```

## Key Design Decisions

1. **Vectorization**: Patches are stored as k² vectors (flattened adjacency matrices) because this format is standard for dictionary learning algorithms.

2. **MCMC Sampling**: Using Markov chains ensures:
   - Efficient sampling from large networks
   - Theoretical guarantees about sampling distribution
   - Ability to sample from specific distributions (weighted by edge counts, etc.)

3. **Folded Homomorphisms**: The `skip_folded_hom` option addresses the tradeoff:
   - Including folded homomorphisms: More samples, captures network properties including degree distribution
   - Excluding folded homomorphisms: Cleaner samples, but slower sampling and potential bias

4. **Flexible Network Types**: Supports both:
   - Regular networks (binary or weighted edges)
   - Tensor networks (edges with multiple color/type dimensions)

## Performance Considerations

- **Sample Size**: Larger sample_size increases computation time linearly
- **Motif Size**: Complexity increases with k² (motif size squared)
- **Skip Folded**: Setting `skip_folded_hom=True` can significantly slow sampling in dense networks
- **Sampling Algorithm**: Different algorithms have different mixing times:
  - Pivot generally fastest
  - Glauber more thorough for some network types
  - IDLA guaranteed injective but potentially slower

## Related Research

This method implements the patch sampling strategy described in:
- Lyu, H., Kureh, Y., Vendrow, J., & Porter, M. A. (2021). "Learning low-rank latent mesoscale structures in networks." arXiv:2102.06984

The approach combines ideas from:
- Graph homomorphism sampling
- Dictionary learning (sparse coding)
- Network motif analysis
- Markov chain Monte Carlo methods
