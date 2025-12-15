# QPINN: Quantum Physics-Informed Neural Networks

A Jupyter notebook implementation comparing **Classical** and **Quantum Physics-Informed Neural Networks (PINNs)** for solving the 2D incompressible Navier–Stokes equations using the cylinder wake benchmark dataset.

---

## 📖 Overview

This project explores the application of hybrid quantum-classical neural networks to physics-informed machine learning. It implements and compares different neural network architectures for solving fluid dynamics problems governed by the Navier–Stokes equations.

### Key Features

- **Classical PINN Implementation**: Fully-connected multilayer perceptron (MLP) with physics-informed loss functions
- **Quantum PINN Implementation**: Hybrid quantum-classical architectures using PennyLane
- **Multiple Quantum Architectures**: Original (simple) and improved (deeper classical layers) variants
- **Comprehensive Visualization**: Animated vorticity comparisons, error metrics, and training diagnostics
- **Benchmark Dataset**: Uses the well-established cylinder wake dataset from Raissi et al.

---

## 🔬 Problem Description

### Cylinder Wake Dataset

The notebook uses the cylinder wake dataset from the seminal PINN paper:

> **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations**  
> *Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Journal of Computational Physics, 378, 686-707.*

#### Problem Setup

- **Flow Type**: 2D, incompressible flow governed by the Navier–Stokes equations
- **Governing Equations**:
  
  $$\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}, \quad \nabla\cdot\mathbf{u}=0$$

#### Non-dimensionalization

| Parameter | Value |
|-----------|-------|
| Cylinder diameter | D = 1 |
| Inflow speed | U∞ = 1 |
| Kinematic viscosity | ν = 0.01 |
| Reynolds number | Re = 100 |

#### Boundary Conditions

| Boundary | Condition |
|----------|-----------|
| **Inflow (left)** | Dirichlet: u = 1, v = 0 |
| **Cylinder surface** | No-slip: u = v = 0 |
| **Top/bottom** | Neumann: ∂ₙu = ∂ₙv = ∂ₙp = 0 |
| **Outflow (right)** | Neumann: ∂ₙu = ∂ₙv = ∂ₙp = 0 |

#### Flow Regime
Laminar vortex shedding (von Kármán street)

#### Dataset Details

| Property | Value |
|----------|-------|
| Spatial domain | x ∈ [1.00, 8.00], y ∈ [-2.00, 2.00] |
| Time interval | t ∈ [0.00, 19.90] |
| Grid (x × y) | 100 × 50 spatial points |
| Points per timestep | 5,000 (N = Nx × Ny) |
| Time steps | 200 |
| Total samples | 1,000,000 space–time points |
| Fields | u(x,y,t), v(x,y,t), p(x,y,t) |

---

## 🏗️ Architecture

### Classical PINN

The classical neural network uses a fully-connected MLP that maps:

$$(x, y, t) \longrightarrow (u, v, p)$$

**Key Features:**
- **Glorot Initialization**: Xavier initialization for stable training
- **First Layer**: Sinusoidal activation `sin(2π(Wx + b))` for multi-scale representation
- **Hidden Layers**: Tanh activation for smooth, differentiable outputs
- **Output Layer**: Linear (no activation)

### Quantum PINN Architectures

#### Architecture 1: Original (Simple)
```
[3] → Dense(n_qubits) → Quantum(n_qubits q × n_layers L) → [3]
```

**Parameter Count:**
$$N_{\text{params}} = n_q(7 + 3n_L) + 3$$

#### Architecture 2: Improved (Deeper Classical)
```
[3] → Dense(16) → Dense(n_qubits) → Quantum(n_qubits q × n_layers L) → Dense(16) → [3]
```

**Features:**
- Deeper classical preprocessing and postprocessing
- Input normalization to [0,1]
- Fourier features for better representation

### Quantum Circuit

The quantum layer uses **StronglyEntanglingLayers** from PennyLane, which applies:
- Parameterized rotation gates
- Entangling CNOT gates
- Multiple layers for expressivity

---

## 📊 Physics-Informed Loss Function

The PINN loss combines data fidelity with PDE residuals:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}}$$

### Data Loss
$$\mathcal{L}_{\text{data}} = \text{MSE}(u_{\text{pred}}, u_{\text{true}}) + \text{MSE}(v_{\text{pred}}, v_{\text{true}}) + \text{MSE}(p_{\text{pred}}, p_{\text{true}})$$

### PDE Residual Loss
Based on the Navier–Stokes equations:
- **x-momentum**: $f = u_t + uu_x + vu_y + \frac{1}{\rho}p_x - \nu(u_{xx} + u_{yy})$
- **y-momentum**: $g = v_t + uv_x + vv_y + \frac{1}{\rho}p_y - \nu(v_{xx} + v_{yy})$
- **Continuity**: $h = u_x + v_y$

$$\mathcal{L}_{\text{PDE}} = \text{MSE}(f) + \text{MSE}(g) + \text{MSE}(h)$$

### Computational Complexity

For a PINN computing Navier-Stokes residuals with batch size B, input dimension d=3, output dimension m=3:

$$T_{\text{PINN}} = O\Big(28B \cdot C_{\text{forward}} \Big)$$

**Breakdown:**
- 1 forward pass for prediction
- 9 forward passes for first derivatives (m outputs × d inputs)
- 18 forward passes for second derivatives (2 Hessians for u, v)

---

## 🛠️ Dependencies

```python
numpy
jax
jax.numpy (jnp)
pennylane (qml)
optax
matplotlib
scipy
```

### Installation

```bash
pip install numpy jax jaxlib pennylane optax matplotlib scipy
```

---

## 📁 Project Structure

```
qpinn/
├── README.md              # This file
├── qpinn.ipynb            # Main Jupyter notebook
└── data/
    └── cylinder_wake.mat  # Benchmark dataset
```

---

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/geet/qpinn.git
   cd qpinn
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Place `cylinder_wake.mat` in the `data/` directory. The dataset can be obtained from [Raissi's PINN repository](https://github.com/maziarraissi/PINNs).

4. **Run the notebook:**
   ```bash
   jupyter notebook qpinn.ipynb
   ```

---

## 📈 Key Functions

### Data Loading & Preprocessing
- `load_cylinder_wake()` - Load the .mat dataset
- `sample_training()` - Random sampling for training data
- `normalize01()` - Input normalization to [0,1]

### Network Architectures
- `init_classical_params()` / `classical_network()` - Classical MLP
- `init_quantum_params_original()` / `quantum_network_original()` - Simple hybrid architecture
- `init_quantum_params_improved()` / `quantum_network_improved()` - Improved hybrid architecture

### Physics-Informed Components
- `residuals_single()` - Compute NS residuals at a single point
- `residuals_batch()` - Vectorized residuals over batch
- `loss_fn()` - Total PINN loss function
- `uvp_batch()` - Batch prediction of (u, v, p)

### Training
- `train_lbfgs()` - L-BFGS optimizer training
- `train_adam()` - Adam optimizer with warmup and decay

### Evaluation & Visualization
- `compute_error_metrics()` - Error metrics computation
- `animate_vorticity_truth_vs_samples()` - Animated vorticity comparison
- `animate_multi_comparison()` - Multi-architecture comparison

---

## 📚 References

1. **Original PINN Paper:**
   > Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.

2. **Glorot Initialization:**
   > Xavier Glorot & Yoshua Bengio (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks.* AISTATS 2010.

3. **Reference Implementation:**
   > [Matt2371/PINN_navier_stokes](https://github.com/Matt2371/PINN_navier_stokes/) (PyTorch)

---

## 📝 License

This project is provided for educational and research purposes.

---

## 👨‍💻 Author

[Your Name]

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
