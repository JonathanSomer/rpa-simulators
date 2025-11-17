import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from rpasim.ode import ODE


def plot_trajectory(
    ode: ODE,
    x0: torch.Tensor,
    T: float,
    n_steps: int = 1000,
    figsize: tuple = (6, 2),
):
    """Plot ODE trajectory from initial state x0 for time T.

    Args:
        ode: ODE instance to simulate
        x0: Initial state tensor
        T: Total time to simulate
        n_steps: Number of time steps
        figsize: Figure size per subplot (default: (6, 3))

    Returns:
        fig, axes: Figure and axes objects
    """
    # Create time points
    t = torch.linspace(0, T, n_steps)

    # Integrate ODE
    trajectory = odeint(ode, x0, t)

    # Create subplots - one per state variable
    n_vars = trajectory.shape[1]
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars))

    # Handle single variable case
    if n_vars == 1:
        axes = [axes]

    # Plot each state variable on its own axis
    for i, ax in enumerate(axes):
        ax.plot(t.numpy(), trajectory[:, i].detach().numpy())
        ax.set_xlabel("time")

        # Use variable name from ODE if available, otherwise use x{i}
        if hasattr(ode, "variable_names") and len(ode.variable_names) == n_vars:
            ylabel = ode.variable_names[i]
        else:
            ylabel = f"x{i}"
        ax.set_ylabel(ylabel)

        sns.despine(ax=ax)

    plt.tight_layout()
    return fig, axes
