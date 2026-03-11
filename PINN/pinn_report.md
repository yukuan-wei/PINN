# Report: Solving an ODE with a Physics-Informed Neural Network (PINN)

**Course presentation (tomorrow)**

**Name:** ____________________  
**Date:** ____________________  
**Repository:** (GitHub/Gitee link) ____________________

---

## 1. Motivation (What is a PINN?)

Physics-Informed Neural Networks (PINNs) are neural networks trained not only to fit data, but also to satisfy governing physical laws written as differential equations.  
Instead of relying on many labeled solution points, a PINN uses the **equation residual** and **boundary/initial conditions** as training signals.

Key idea: approximate the unknown solution with a neural network $u_\theta(x)$, and train $\theta$ so that:

- the differential equation is satisfied in the domain (physics loss)
- the boundary/initial conditions are satisfied (constraint loss)

---

## 2. Problem statement

We solve a simple boundary-value ODE on \([0,1]\):

\[
u''(x) + \pi^2 u(x) = 0,\quad x\in[0,1]
\]

Boundary conditions:

\[
u(0)=0,\qquad u(1)=0
\]

For this setup, the exact solution is:

\[
u(x)=0
\]

This problem is chosen for a clean classroom demonstration because the correctness is easy to verify.

---

## 3. PINN formulation

### 3.1 Neural network approximation

We represent the solution as a fully connected neural network:

$$
u_\theta(x) \approx u(x)
$$

with $x$ as input and $u_\theta(x)$ as output.  
The activation function used is $\tanh(\cdot)$.

### 3.2 Physics (residual) loss

Define the ODE residual:

$$
r_\theta(x) = u_\theta''(x) + \pi^2 u_\theta(x)
$$

We sample **collocation points** $\{x_f^i\}_{i=1}^{N_f}$ in the interior and minimize:

$$
\mathcal{L}_f = \frac{1}{N_f}\sum_{i=1}^{N_f} \left( r_\theta(x_f^i) \right)^2
$$

### 3.3 Boundary loss

We enforce boundary conditions at $x=0$ and $x=1$:

$$
\mathcal{L}_b = \left(u_\theta(0)-0\right)^2 + \left(u_\theta(1)-0\right)^2
$$

### 3.4 Total loss

$$
\mathcal{L} = \mathcal{L}_f + \mathcal{L}_b
$$

### 3.5 Automatic differentiation

The derivatives $u_\theta'(x)$ and $u_\theta''(x)$ are computed by **automatic differentiation** in PyTorch, which avoids manual derivative coding.

---

## 4. Implementation details

### 4.1 Code organization

The main implementation is in:

- `pinn_ode.py`

Key parts:

- Network definition (`PINN` class)
- Residual computation (`pinn_residual`)
- Training loop (Adam optimizer)
- Evaluation + plot saved as `pinn_ode_solution.png`

### 4.2 Hyperparameters (example)

- Domain: $[0,1]$
- Collocation points: $N_f = 100$
- Boundary points: $N_b = 2$
- Network: 1–20–20–20–1
- Optimizer: Adam, learning rate $10^{-3}$
- Epochs: 5000

These can be adjusted if training is slow or if a smoother curve is needed.

---

## 5. Results

After training, we evaluate the network on a dense test grid and compare with the exact solution.

Expected outcome:

- The predicted $u_\theta(x)$ is close to 0 for all $x\in[0,1]$
- The residual loss decreases during training

Insert the figure here:

- `pinn_ode_solution.png`

---

## 6. Common issues and how I fixed them (Windows/Conda)

### 6.1 Why environments matter

On Windows, scientific libraries (NumPy/PyTorch) depend on native DLLs (e.g., MKL/OpenMP).  
If packages are installed in the wrong environment, Python may fail to find the correct DLLs at runtime.

### 6.2 Typical error

An example error is:

- `Intel MKL FATAL ERROR: cannot load mkl_vml_avx2.1.dll`

### 6.3 Fix

I solved it by:

- creating a clean conda environment (`pinn-env`)
- activating it correctly
- installing dependencies inside the activated environment

This ensures the Python interpreter, packages, and DLL search paths are consistent.

---

## 7. Conclusion and future work

### Conclusion

This example demonstrates the key concept of PINNs:

- a neural network can be trained to satisfy differential equations using physics-based losses

### Future work

Possible extensions:

- Use a more meaningful non-trivial solution (e.g., specify a different boundary condition)
- Solve a PDE (heat equation / Burgers’ equation / Poisson equation)
- Study training stability and loss balancing

---

## Appendix: How to run

In Anaconda Prompt:

```bash
conda create -n pinn-env python=3.10 -y
conda activate pinn-env
pip install -r requirements.txt
python pinn_ode.py
```

