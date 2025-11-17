import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_env_trajectory(env, figsize: tuple = (6, 2)):
    """Plot environment trajectory and rewards.

    Args:
        env: DifferentiableEnv instance with trajectory data
        figsize: Figure size per subplot (default: (6, 2))

    Returns:
        fig, axes: Figure and axes objects
    """
    # Get full trajectory
    times, states, rewards = env.get_trajectory()

    # Create subplots - one per state variable + one for rewards
    n_vars = states.shape[1]
    n_plots = n_vars + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots))

    # Handle single variable case
    if n_plots == 1:
        axes = [axes]

    # Plot each state variable on its own axis
    for i in range(n_vars):
        axes[i].plot(times.detach().numpy(), states[:, i].detach().numpy())
        axes[i].set_xlabel("time")
        axes[i].set_ylabel(f"x{i}")
        sns.despine(ax=axes[i])

    # Plot rewards on the last axis
    axes[-1].plot(times.detach().numpy(), rewards.detach().numpy())
    axes[-1].set_xlabel("time")
    axes[-1].set_ylabel("reward")
    sns.despine(ax=axes[-1])

    plt.tight_layout()
    return fig, axes
