# QPINN: Quantum Physics-Informed Neural Networks

Implementation and comparison of classical and quantum physics-informed neural networks for solving the 2D incompressible Navier-Stokes equations. Uses the cylinder wake benchmark dataset from Raissi et al. (2019).

## Overview

This project investigates hybrid quantum-classical architectures for physics-informed machine learning. The implementation compares classical multilayer perceptrons against quantum circuit-based models on a fluid dynamics benchmark problem. Results show that quantum PINNs achieve significant parameter reduction (79-311 vs 2000+) at the cost of training speed and solution accuracy.

## Problem Statement

### Dataset
Flow past a circular cylinder at Reynolds number Re = 100, taken from the [seminal PINN paper](https://doi.org/10.1016/j.jcp.2018.10.045). The flow exhibits laminar vortex shedding (von Kármán vortex street).

**Spatial-temporal domain:**
- Spatial: x ∈ [1, 8], y ∈ [-2, 2] (100 × 50 grid)
- Temporal: t ∈ [0, 19.9] (200 timesteps)
- Total: 1,000,000 space-time points

**Fields:** Velocity components (u, v) and pressure (p)

### Governing Equations

2D incompressible Navier-Stokes equations:

$$\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$

$$\nabla \cdot \mathbf{u} = 0$$

where ν = 0.01 is the kinematic viscosity.

**Boundary conditions:**
- Inflow (left): u = 1, v = 0 (Dirichlet)
- Cylinder surface: u = v = 0 (no-slip)
- Top/bottom walls: ∂ₙu = ∂ₙv = ∂ₙp = 0 (Neumann)
- Outflow (right): ∂ₙu = ∂ₙv = ∂ₙp = 0 (Neumann)

## Model Architectures

All models map input coordinates to flow fields: `(x, y, t) → (u, v, p)`

### Classical PINN

Fully-connected MLP with physics-informed training.

**Architecture details:**
- Input layer: Sinusoidal activation `sin(2π(Wx + b))` for multi-scale feature extraction
- Hidden layers: Tanh activation (smooth, differentiable for PDE residual computation)
- Initialization: Glorot/Xavier uniform
- Output: Linear projection to (u, v, p)

**Configurations:**
- 5-layer variant: 2,128 parameters
- 8-layer variant: 4,078 parameters

### Quantum PINN Variants

Hybrid quantum-classical architectures using PennyLane quantum circuits.

**Quantum circuit:** `StronglyEntanglingLayers` consisting of parameterized single-qubit rotations and entangling CNOT gates arranged in layers.

**Architecture 1 (Original):**
```
Input(3) → Dense(n_qubits) → QuantumCircuit(n_qubits, n_layers) → Output(3)
```
- Parameters: 79
- Minimal classical preprocessing/postprocessing

**Architecture 2 (Improved):**
```
Input(3) → Dense(16) → Dense(n_qubits) → QuantumCircuit(n_qubits, n_layers) → Dense(16) → Output(3)
```
- Parameters: 311
- Input normalization to [0, 1]
- Deeper classical layers for feature transformation

## Physics-Informed Loss

The loss function combines data fidelity with PDE residual minimization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}}$$

**Data loss:**
$$\mathcal{L}_{\text{data}} = \text{MSE}(u_{\text{pred}}, u_{\text{true}}) + \text{MSE}(v_{\text{pred}}, v_{\text{true}}) + \text{MSE}(p_{\text{pred}}, p_{\text{true}})$$

**PDE residual loss:**

For each space-time point, compute residuals of:
- x-momentum: $f = u_t + uu_x + vu_y + \frac{1}{\rho}p_x - \nu(u_{xx} + u_{yy})$
- y-momentum: $g = v_t + uv_x + vv_y + \frac{1}{\rho}p_y - \nu(v_{xx} + v_{yy})$
- Continuity: $h = u_x + v_y$

$$\mathcal{L}_{\text{PDE}} = \text{MSE}(f) + \text{MSE}(g) + \text{MSE}(h)$$

**Computational cost:** Each batch requires 28 forward passes:
- 1 for prediction
- 9 for first-order derivatives (∂u/∂x, ∂u/∂y, ∂u/∂t, ∂v/∂x, ...)
- 18 for second-order derivatives (∂²u/∂x², ∂²u/∂y², ∂²v/∂x², ∂²v/∂y²)

## Implementation

Built with JAX for automatic differentiation and hardware acceleration.

**Key JAX features:**
- `jax.grad()` / `jax.value_and_grad()`: Exact gradient computation for PDE residuals and backpropagation
- `jax.vmap()`: Vectorized batch processing of residual calculations
- `jax.jit()`: XLA compilation for training loop and loss functions
- Integration with Optax (Adam optimizer) and SciPy (L-BFGS-B)

**Quantum backend:** PennyLane with JAX interface for differentiable quantum circuits.

## Experimental Results

All models trained on randomly sampled points using Adam (warmup + decay) followed by L-BFGS-B fine-tuning.

| Model | Parameters | Training Time (s) | Final Loss | Data Loss | PDE Loss |
|-------|------------|-------------------|------------|-----------|----------|
| Classical (8 layers) | 4,078 | 71.46 | 2.38e-03 | 1.22e-03 | 1.16e-03 |
| Classical (5 layers) | 2,128 | 43.88 | 3.35e-03 | 1.81e-03 | 1.54e-03 |
| Quantum (improved) | 311 | 372.27 | 2.89e-02 | 2.20e-02 | 6.83e-03 |
| Quantum (original) | 79 | 327.90 | 1.03e-01 | 1.00e-01 | 2.35e-03 |

**Observations:**
- Classical networks achieve superior accuracy and training efficiency
- Quantum variants provide 6-50× parameter reduction but suffer from slower convergence and higher final loss
- The improved quantum architecture partially closes the performance gap through deeper classical preprocessing

## Running the Code

Clone and install dependencies:
```bash
pip install numpy jax jaxlib pennylane optax matplotlib scipy
```

Open `qpinn.ipynb` and run all cells. The notebook will:
1. Load the cylinder wake dataset
2. Train all 4 architectures
3. Generate comparison plots and animations

Dataset should be at `data/cylinder_wake.mat` (download from the Raissi paper repo if needed).

## Main Functions

**Data:**
- `load_cylinder_wake()` - loads the `.mat` file
- `sample_training()` - random sampling for training
- `normalize01()` - normalizes inputs to [0,1]

**Models:**
- `classical_network()` - MLP forward pass
- `quantum_network_original()` - simple quantum hybrid
- `quantum_network_improved()` - improved quantum hybrid

**Physics:**
- `residuals_single()` / `residuals_batch()` - compute NS residuals
- `loss_fn()` - total PINN loss (data + PDE)

**Training:**
- `train_adam()` - Adam with learning rate schedule
- `train_lbfgs()` - L-BFGS-B fine-tuning

**Visualization:**
- `animate_vorticity_truth_vs_samples()` - single model comparison
- `animate_multi_comparison()` - all models side-by-side

## References

- Raissi et al. (2019) - Original PINN paper and cylinder wake dataset
- [Matt2371/PINN_navier_stokes](https://github.com/Matt2371/PINN_navier_stokes/) - PyTorch reference implementation  
- Glorot & Bengio (2010) - Xavier initialization

## License

MIT
