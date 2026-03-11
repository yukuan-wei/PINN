# PINN ODE Example

This folder contains a simple Physics-Informed Neural Network (PINN) example that solves
the boundary-value problem

u''(x) + π² u(x) = 0,  x ∈ [0, 1],  u(0) = 0,  u(1) = 0.

The exact solution is u(x) = 0. The PINN learns an approximation that satisfies the
differential equation and boundary conditions.

## 1. Create a new Anaconda environment

```bash
conda create -n pinn-env python=3.10 -y
conda activate pinn-env
pip install -r requirements.txt
```

If installing `torch` fails, visit the official PyTorch (CPU) install page and follow the
command for Windows.

## 2. Run the PINN code

```bash
conda activate pinn-env
python pinn_ode.py
```

This will:

- Train a small fully connected neural network as a PINN.
- Print the training loss every 500 epochs.
- Generate a plot comparing the PINN solution with the exact solution and save it as
  `pinn_ode_solution.png`.

## 3. Report (PDF) structure

Create a PDF document (e.g. using Word, WPS, or LaTeX) with content similar to:

1. **Title**: Solving an ODE using Physics-Informed Neural Networks (PINNs).
2. **Problem Statement**: Describe the ODE and boundary conditions.
3. **PINN Idea**:
   - Neural network approximates u(x).
   - Physics loss: residual of u''(x) + π² u(x) = 0.
   - Boundary loss: enforce u(0) = 0 and u(1) = 0.
4. **Network and Training**: Briefly describe architecture, collocation points, optimizer.
5. **Results**: Insert `pinn_ode_solution.png` and compare with the exact solution.
6. **Discussion**: Short comments on advantages, limitations, and possible extensions.

Save the file as `pinn_ode_explanation.pdf` in this same folder.

## 4. Upload to GitHub or Gitee

In this folder, you can use git to upload the code and PDF:

```bash
git init
git add .
git commit -m "Add PINN ODE example and documentation"

# GitHub:
git remote add origin https://github.com/yourname/pinn-ode-example.git

# or Gitee:
# git remote add origin https://gitee.com/yourname/pinn-ode-example.git

git branch -M main
git push -u origin main
```

## 5. Suggested classroom demonstration flow (English)

1. **Introduction**  
   “Today I will show how to use a Physics-Informed Neural Network, or PINN, to solve a
   simple differential equation.”

2. **Problem setup**  
   “We consider the ordinary differential equation
   u''(x) + π² u(x) = 0 on [0, 1], with boundary conditions u(0) = 0 and u(1) = 0.
   The exact solution is u(x) = 0, which makes it easy to check whether our PINN works.”

3. **PINN method**  
   “We build a fully connected neural network u_θ(x) to approximate u(x).  
   The physics loss is the mean squared residual of the ODE.  
   The boundary loss enforces the conditions at x = 0 and x = 1.  
   We use automatic differentiation in PyTorch to compute derivatives.”

4. **Code and training**  
   “In the code, we define the network, the residual function, and the training loop.  
   Then we train the model and plot both the PINN solution and the exact solution.”

5. **Results and discussion**  
   “The plot shows that the PINN solution is very close to u(x) = 0.  
   This demonstrates that the network has learned to satisfy both the differential
   equation and the boundary conditions. In more complex problems, PINNs can handle
   higher-dimensional PDEs, but training can be more challenging.”

