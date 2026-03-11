import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(nn.Tanh())
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


def pinn_residual(x, model):
    """Residual of u''(x) + pi^2 u(x) = 0."""
    x = x.clone().detach().requires_grad_(True)
    u = model(x)

    du_dx = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = d2u_dx2 + (np.pi**2) * u
    return residual


def main():
    # Training points in [0, 1]
    N_f = 100  # collocation points for PDE residual
    N_b = 2  # boundary points

    x_f = torch.linspace(0.0, 1.0, N_f).view(-1, 1).to(device)

    # Boundary points: x = 0, 1
    x_b = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)
    u_b = torch.zeros_like(x_b).to(device)  # u(0) = u(1) = 0

    layers = [1, 20, 20, 20, 1]
    model = PINN(layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual loss
        r_f = pinn_residual(x_f, model)
        loss_f = torch.mean(r_f**2)

        # Boundary loss
        u_pred_b = model(x_b)
        loss_b = torch.mean((u_pred_b - u_b) ** 2)

        loss = loss_f + loss_b

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Total Loss: {loss.item():.4e}, "
                f"Residual Loss: {loss_f.item():.4e}, Boundary Loss: {loss_b.item():.4e}"
            )

    # Evaluation
    x_test = torch.linspace(0.0, 1.0, 200).view(-1, 1).to(device)
    u_pred = model(x_test).detach().cpu().numpy()

    x_test_np = x_test.detach().cpu().numpy()
    u_exact = np.zeros_like(x_test_np)  # exact solution u(x) = 0

    plt.figure(figsize=(6, 4))
    plt.plot(x_test_np, u_pred, label="PINN solution")
    plt.plot(x_test_np, u_exact, "k--", label="Exact solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("PINN solution of u'' + π²u = 0 with u(0)=u(1)=0")
    plt.tight_layout()
    plt.savefig("pinn_ode_solution.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

