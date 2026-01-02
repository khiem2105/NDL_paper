# Detailed Explanation of Sampling Algorithms and update_hom_get_meso_patch Method

## Overview

This document provides a comprehensive explanation of the MCMC sampling algorithms used in the Network Dictionary Learning (NDL) framework and the `update_hom_get_meso_patch` method that orchestrates them.

## Table of Contents

1. [The update_hom_get_meso_patch Method](#the-update_hom_get_meso_patch-method)
2. [Sampling Algorithms](#sampling-algorithms)
   - [Glauber Dynamics](#1-glauber-dynamics-glauber_gen_update)
   - [Pivot Chain](#2-pivot-chain-pivot_update)
   - [IDLA (Internal Diffusion Limited Aggregation)](#3-idla-internal-diffusion-limited-aggregation)
   - [Pivot Injective](#4-pivot-injective-pivot_inj)
3. [Helper Methods](#helper-methods)
4. [Performance Comparison](#performance-comparison)

---

## The update_hom_get_meso_patch Method

### Purpose

The `update_hom_get_meso_patch` method is the core engine that:
1. **Updates** the embedding of a motif into the network using MCMC
2. **Extracts** the induced subgraph (patch) based on the embedding
3. **Averages** patches over multiple iterations (if requested)
4. **Tracks** edge folding information (optional)

### Method Signature

```python
def update_hom_get_meso_patch(self, B, emb, iterations=1, 
                              sampling_alg='glauber', 
                              verbose=0, 
                              omit_folded_edges=False):
```

### Algorithm Walkthrough

#### Phase 1: Initialization

```python
N = self.G                    # The network
emb2 = emb                    # Current embedding
k = B.shape[0]                # Motif size

# Initialize patch storage
hom_mx2 = np.zeros([k, k])    # For regular networks
if self.if_tensor_ntwk:
    hom_mx2 = np.zeros([k, k, N.color_dim])  # For tensor networks

nofolding_ind_mx = np.zeros([k, k])  # Tracks non-folded edges
```

**Purpose**: Set up data structures to accumulate patch information over multiple iterations.

#### Phase 2: MCMC Update Loop

```python
for i in range(iterations):
    # Update embedding using selected algorithm
    if sampling_alg == 'glauber':
        emb2 = self.glauber_gen_update(B, emb2)
    elif sampling_alg == 'pivot':
        emb2 = self.Pivot_update(emb2, if_inj=False)
    elif sampling_alg == 'idla':
        H = None
        while H is None:
            H = N.k_node_IDLA_subgraph(k=k, center=None)
        emb2 = H.nodes()
    elif sampling_alg == 'pivot_inj':
        emb2 = self.Pivot_update(emb2, if_inj=True)
```

**Purpose**: Update the embedding using the specified MCMC algorithm. Each algorithm has different mixing properties and theoretical guarantees.

#### Phase 3: Induced Subgraph Formation

```python
# Form induced graph H = homomorphic copy of motif
H = NNetwork()
for q in range(k):
    for r in range(k):
        edge = [emb2[q], emb2[r]]
        if B[q, r] > 0:  # Edge exists in motif
            H.add_edge(edge=edge, weight=1)
```

**Purpose**: Create a helper graph `H` that represents the motif structure mapped onto the network. This helps detect "folded" edges.

#### Phase 4: Patch Extraction

For **regular networks**:

```python
a2 = np.zeros([k, k])  # Initialize patch
for q in range(k):
    for r in range(k):
        if not self.if_wtd_network or N.has_edge(emb2[q], emb2[r]) == 0:
            # Binary edge (0 or 1)
            a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
        else:
            # Weighted edge
            a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])
```

For **tensor networks** (colored edges):

```python
a2 = np.zeros([k, k, N.color_dim])
for q in range(k):
    for r in range(k):
        if N.has_edge(emb2[q], emb2[r]) == 0:
            a2[q, r, :] = np.zeros(N.color_dim)
        else:
            a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
```

**Folding Detection** (if `omit_folded_edges=True`):

```python
# An edge is "folded" if:
# 1. It doesn't exist in the motif (B[q,r] + B[r,q] == 0)
# 2. BUT it exists in the induced graph H (same network node appears multiple times in embedding)

if not (B[q, r] + B[r, q] == 0 and H.has_edge(emb2[q], emb2[r]) == 1):
    a2[q, r] = ...  # Include this edge
    nofolding_ind_mx[q, r] = 1
```

**Example of Folding**:
- Motif: Linear chain A—B—C
- Embedding: emb = [node_1, node_2, node_1] (A→1, B→2, C→1)
- Edge (A,C) doesn't exist in motif, but (1,1) might exist in network as a self-loop
- This is a "folded" edge that appeared due to the non-injective embedding

#### Phase 5: Running Average

```python
hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)
```

**Purpose**: Compute a running average of patches over iterations. This reduces variance in the sampled patches.

### Return Values

```python
if omit_folded_edges:
    return hom_mx2, emb2, nofolding_ind_mx
else:
    return hom_mx2, emb2
```

- `hom_mx2`: The averaged patch (k×k matrix or k×k×color_dim tensor)
- `emb2`: The final embedding after all iterations
- `nofolding_ind_mx`: Binary indicator of non-folded edges (optional)

---

## Sampling Algorithms

### 1. Glauber Dynamics (`glauber_gen_update`)

#### Overview

Glauber dynamics updates the embedding by randomly selecting one position in the motif and resampling it conditioned on all other positions. This maintains the homomorphism property (edges in the motif are mapped to edges in the network).

#### Algorithm Details

**Step 1: Select Position to Update**

```python
j = np.random.choice(np.arange(0, k))  # Random position in [0, k-1]
```

**Step 2: Find Constraints**

```python
nbh_in = self.indices(B[:, j], lambda x: x == 1)   # Incoming neighbors in motif
nbh_out = self.indices(B[j, :], lambda x: x == 1)  # Outgoing neighbors in motif
```

These are the positions in the motif that are connected to position `j`.

**Step 3: Compute Valid Candidates**

For **unweighted networks**:

```python
cmn_nbs = N.nodes(is_set=True)  # Start with all nodes

# Intersect with neighbors of each connected position
for r in nbh_in:
    nbs_r = N.neighbors(emb[r])
    cmn_nbs = cmn_nbs & nbs_r

for r in nbh_out:
    nbs_r = N.neighbors(emb[r])
    cmn_nbs = cmn_nbs & nbs_r
```

**Purpose**: Find nodes that maintain the homomorphism property. The new value `emb[j]` must be connected to all the network nodes that `emb[j]` should be connected to according to the motif structure.

For **weighted networks**:

```python
# Compute distribution weighted by edge weights
dist = np.ones(len(cmn_nbs))
for v in range(len(cmn_nbs)):
    for r in nbh_in:
        dist[v] *= abs(N.get_edge_weight(emb[r], cmn_nbs[v]))
    for r in nbh_out:
        dist[v] *= abs(N.get_edge_weight(cmn_nbs[v], emb[r]))
dist = dist / np.sum(dist)

idx = np.random.choice(np.arange(len(cmn_nbs)), p=dist)
emb[j] = cmn_nbs[idx]
```

**Purpose**: Weight candidates by the product of edge weights. This makes the sampling favor stronger connections.

**Step 4: Update or Reject**

```python
if len(cmn_nbs) > 0:
    y = np.random.choice(np.asarray(cmn_nbs))
    emb[j] = y
else:
    emb[j] = np.random.choice(N.nodes())  # Fallback (shouldn't happen)
    print('Glauber move rejected')
```

#### Mathematical Properties

- **Stationarity**: Converges to uniform distribution over valid homomorphisms
- **Detailed Balance**: Satisfies detailed balance for the target distribution
- **Mixing Time**: O(k² log(n)) for many networks, where k = motif size, n = network size
- **Maintains Homomorphism**: Always produces valid homomorphisms

#### Advantages

- Theoretically well-understood
- Maintains exact homomorphism property
- Works well for small to medium motifs

#### Disadvantages

- Can be slow to mix for large motifs
- May get stuck in local regions of the network
- Requires intersection operations which can be expensive

---

### 2. Pivot Chain (`Pivot_update`)

#### Overview

The pivot chain designates one position (the "pivot") in the motif, moves it via random walk, then resamples the entire motif from the new pivot position. This leads to faster mixing by making larger moves in the embedding space.

#### Algorithm Details

**Step 1: Update Pivot via Random Walk**

```python
x0 = emb[0]  # Current pivot location
x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)
```

The `RW_update` method:

```python
def RW_update(self, x, Pivot_exact_MH_rule=False):
    nbs_x = list(N.neighbors(x))
    
    if len(nbs_x) > 0:
        y = np.random.choice(nbs_x)  # Propose move to neighbor
        nbs_y = list(N.neighbors(y))
        
        # Metropolis-Hastings acceptance probability
        prob_accept = min(1, len(nbs_x) / len(nbs_y))
        
        if np.random.rand() > prob_accept:
            y = x  # Reject move
    else:
        y = np.random.choice(N.nodes())  # Handle isolated node
    
    return y
```

**Purpose**: Perform one step of a Metropolis-Hastings random walk with uniform stationary distribution. This moves the pivot to a new location in the network.

**Optional Exact MH Rule**:

```python
if Pivot_exact_MH_rule:
    a = N.count_k_step_walks(y, radius=length)
    b = N.count_k_step_walks(x, radius=length)
    prob_accept = min(1, a * len(nbs_x) / (b * len(nbs_y)))
```

**Purpose**: Corrects for the fact that different nodes have different numbers of possible motif embeddings. Makes the chain sample from the exact uniform distribution over homomorphisms (not just nodes).

**Step 2: Resample Motif from New Pivot**

```python
B = self.path_adj(0, len(emb)-1)  # Create motif structure
emb_new = self.tree_sample(B, x0)  # Sample motif rooted at new pivot
```

The `tree_sample` method:

```python
def tree_sample(self, B, x):
    emb = [x]  # Start with pivot
    
    for i in range(1, k):
        j = self.find_parent(B, i)  # Parent in motif
        nbs_j = list(N.neighbors(emb[j]))
        if len(nbs_j) > 0:
            y = np.random.choice(nbs_j)
            emb.append(y)
    
    return emb
```

**Purpose**: Build the motif by traversing the tree structure, sampling a random neighbor at each step.

#### Mathematical Properties

- **Stationarity**: Converges to uniform distribution over valid homomorphisms (with exact MH rule)
- **Mixing Time**: Often O(k log(n)), faster than Glauber for large motifs
- **Large Moves**: Can jump to distant parts of the network
- **Maintains Homomorphism**: Always produces valid homomorphisms

#### Advantages

- Faster mixing than Glauber for many networks
- Makes larger moves in embedding space
- Simple and efficient to implement
- Can escape local regions of the network

#### Disadvantages

- May produce correlated samples if not run long enough
- The "resample all from pivot" strategy may not be optimal for all motif structures
- Without exact MH rule, doesn't sample exact target distribution

---

### 3. IDLA (Internal Diffusion Limited Aggregation)

#### Overview

IDLA samples a k-node subgraph by performing a random walk from a center node and adding nodes as they are visited. This guarantees that all nodes in the embedding are distinct (injective).

#### Algorithm Details

```python
H = None
while H is None:
    H = N.k_node_IDLA_subgraph(k=k, center=None)
emb2 = H.nodes()
```

**The k_node_IDLA_subgraph method** (in NNetwork class):

1. **Initialize**: Start from a random center node (or specified center)
2. **Grow Cluster**: Perform random walks from the boundary of the current cluster
3. **Add Nodes**: When a walk visits a new node, add it to the cluster
4. **Stop**: Continue until k distinct nodes are collected

**Pseudo-code**:
```python
def k_node_IDLA_subgraph(G, k, center=None):
    if center is None:
        center = random.choice(G.nodes())
    
    cluster = {center}
    boundary = {center}
    
    while len(cluster) < k:
        # Pick random node from boundary
        start = random.choice(list(boundary))
        
        # Random walk until hitting new node
        current = start
        while current in cluster:
            current = random.choice(list(G.neighbors(current)))
        
        # Add new node to cluster
        cluster.add(current)
        boundary.add(current)
    
    return G.subgraph(cluster)
```

#### Mathematical Properties

- **Injective**: Always produces distinct nodes
- **Connected**: Tends to produce connected subgraphs
- **Spatial Bias**: Favors nodes close to the center
- **No Homomorphism Constraint**: Doesn't respect motif structure

#### Advantages

- Guarantees injective embeddings
- Fast sampling
- Produces spatially coherent subgraphs
- No need to check for valid homomorphisms

#### Disadvantages

- Doesn't maintain homomorphism property
- Spatial bias (not uniform over all k-node subgraphs)
- May not sample subgraphs that don't form connected components
- The embedding doesn't respect the motif structure B

#### When to Use

Use IDLA when:
- You want guaranteed distinct nodes
- The motif structure is less important than having a k-node subgraph
- You want faster sampling at the cost of not respecting homomorphisms

---

### 4. Pivot Injective (`pivot_inj`)

#### Overview

Combines the pivot chain with IDLA to get the benefits of both: faster mixing from pivot updates and guaranteed distinct nodes from IDLA.

#### Algorithm Details

```python
x0 = emb[0]  # Current pivot
x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)

# Use IDLA to sample k distinct nodes centered at new pivot
H = None
while H is None:
    H = self.G.k_node_IDLA_subgraph(k=len(emb), center=x0)
    if H is None:
        x0 = self.RW_update(x0, Pivot_exact_MH_rule=self.Pivot_exact_MH_rule)

emb_new = H.nodes()
```

#### Mathematical Properties

- **Injective**: Guaranteed distinct nodes
- **Faster Mixing**: Pivot updates help explore the network
- **Spatial Coherence**: IDLA creates locally coherent subgraphs
- **No Homomorphism Constraint**: Still doesn't respect motif structure

#### Advantages

- Combines benefits of pivot (fast mixing) and IDLA (injectivity)
- Good for sampling diverse injective subgraphs
- The pivot update helps explore different regions

#### Disadvantages

- Doesn't maintain homomorphism property
- More complex than either pivot or IDLA alone
- May fail to find IDLA subgraph (hence the retry loop)

---

## Helper Methods

### tree_sample

**Purpose**: Sample an initial embedding of a tree motif.

**Algorithm**:
1. Start with a random root node
2. For each node in the tree (in depth-first order):
   - Find its parent in the tree
   - Sample a random neighbor of the parent's image
   - This becomes the current node's image

### RW_update

**Purpose**: Perform one step of Metropolis-Hastings random walk with uniform stationary distribution.

**Why MH?**: Different nodes have different degrees. Without MH correction, the random walk would have a degree-weighted stationary distribution. MH ensures uniform stationary distribution.

### find_parent

**Purpose**: Find the parent of a node in a tree motif (used for tree_sample).

---

## Performance Comparison

| Algorithm | Mixing Time | Maintains Homomorphism | Injective | Best Use Case |
|-----------|-------------|------------------------|-----------|---------------|
| **Glauber** | Slow for large k | ✓ Yes | ✗ No | Small motifs, exact homomorphisms required |
| **Pivot** | Fast | ✓ Yes | ✗ No | Large motifs, faster sampling, homomorphisms required |
| **IDLA** | Very Fast | ✗ No | ✓ Yes | Need distinct nodes, motif structure less important |
| **Pivot Injective** | Fast | ✗ No | ✓ Yes | Need distinct nodes with good mixing |

### Recommendations

**For Dictionary Learning** (`train_dict`):
- Use **Pivot** for best balance of speed and homomorphism property
- Use **Glauber** if you need exact theoretical guarantees
- Set `skip_folded_hom=False` to allow non-injective samples (captures degree distribution)

**For Network Reconstruction** (`reconstruct_network`):
- Use **Pivot** for standard reconstruction
- Use **IDLA** or **Pivot Injective** with `skip_folded_hom=True` if you only want distinct-node patches

**For Network Analysis**:
- Use **Glauber** for small motifs (k ≤ 10)
- Use **Pivot** for larger motifs (k > 10)
- Use **IDLA** if injectivity is crucial

---

## Example: Comparing Algorithms

```python
import numpy as np
from NNetwork import NNetwork
from ndl import Network_Reconstructor

# Create a test network
G = NNetwork()
# ... add edges to G ...

# Initialize reconstructor
reconstructor = Network_Reconstructor(G, n_components=50, k2=5)

# Create a path motif
B = reconstructor.path_adj(k1=0, k2=5)  # 6-node path
x0 = np.random.choice(list(G.vertices()))
emb = reconstructor.tree_sample(B, x0)

# Compare sampling algorithms
algorithms = ['glauber', 'pivot', 'idla', 'pivot_inj']
patches = {}

for alg in algorithms:
    X, emb_final = reconstructor.get_patches(
        B=B,
        emb=emb,
        sample_size=100,
        sampling_alg=alg,
        skip_folded_hom=False
    )
    patches[alg] = X
    
    # Check how many embeddings have distinct nodes
    # (only meaningful for glauber and pivot)
    if alg in ['glauber', 'pivot']:
        # Would need to track embeddings to compute this
        pass

# Analyze diversity and mixing
# patches['glauber'] - most theoretically sound
# patches['pivot'] - fastest while maintaining homomorphisms
# patches['idla'] - all distinct nodes, but no homomorphism
# patches['pivot_inj'] - distinct nodes with better mixing
```

---

## Theoretical Background

### Homomorphism Sampling

A **graph homomorphism** from motif F to network G is a mapping φ: V(F) → V(G) such that if (u,v) is an edge in F, then (φ(u), φ(v)) is an edge in G.

**Why it matters**: Homomorphisms preserve the structure of the motif in the network. This is crucial for:
- Finding specific patterns (triangles, chains, stars)
- Learning dictionaries that represent real network structures
- Maintaining interpretability of learned patterns

### MCMC Fundamentals

**Goal**: Sample uniformly from the set of all valid homomorphisms.

**Challenge**: The number of homomorphisms can be exponential, making enumeration infeasible.

**Solution**: Use Markov Chain Monte Carlo (MCMC) to sample without enumeration:
1. Define a Markov chain whose stationary distribution is uniform over homomorphisms
2. Run the chain for long enough to approximately reach stationarity
3. The current state is approximately a uniform sample

**Mixing Time**: How long to run before samples are approximately uniform.

### Why Different Algorithms?

Different algorithms trade off:
- **Theoretical guarantees** vs. **Practical speed**
- **Homomorphism preservation** vs. **Injectivity**
- **Local updates** vs. **Global moves**
- **Simplicity** vs. **Flexibility**

---

## Summary

- **`update_hom_get_meso_patch`** orchestrates MCMC sampling and patch extraction
- **Glauber dynamics** provides theoretically sound homomorphism sampling with local updates
- **Pivot chain** achieves faster mixing with global updates
- **IDLA** guarantees distinct nodes but sacrifices homomorphism property
- **Pivot Injective** combines pivot's mixing with IDLA's injectivity

The choice of algorithm depends on your specific needs: homomorphism accuracy, sampling speed, or node distinctness.
