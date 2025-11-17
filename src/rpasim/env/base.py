import torch
from typing import Callable, Tuple, Dict
from copy import deepcopy
from torchdiffeq import odeint
from rpasim.ode import ODE


class DifferentiableEnv:
    """Differentiable environment for ODE control.

    Implements the gymnasium Env API without inheritance.

    Action space: (ODE, time) tuples
    Observation space: (ODE, state) tuples
    """

    def __init__(
        self,
        initial_ode: ODE,
        reward_fn: Callable[[torch.Tensor], float],
        initial_state: torch.Tensor,
        time_horizon: float,
        n_reward_steps: int,
    ):
        """Initialize the differentiable environment.

        Args:
            initial_ode: Initial ODE instance
            reward_fn: Function mapping state -> reward
            initial_state: Initial state tensor
            time_horizon: Maximum time for the environment
            n_reward_steps: Number of steps for reward computation grid
        """
        self.initial_ode = initial_ode
        self.reward_fn = reward_fn
        self.initial_state = initial_state
        self.time_horizon = time_horizon
        self.n_reward_steps = n_reward_steps
        self.reward_dt = time_horizon / n_reward_steps

        # Precompute reward grid
        self.reward_grid = torch.arange(0, time_horizon + self.reward_dt, self.reward_dt)

        # Current state
        self.current_ode = None
        self.current_state = None
        self.current_time = 0.0
        self.trajectory_segments = []
        self.time_segments = []
        self.reward_segments = []

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Tuple[ODE, torch.Tensor], Dict]:
        """Reset environment to initial state.

        Returns:
            observation: (ODE copy, state)
            info: Empty dict
        """
        # Reset to initial state
        self.current_ode = deepcopy(self.initial_ode)
        self.current_state = self.initial_state.clone()
        self.current_time = 0.0

        # Initialize empty trajectory
        self.trajectory_segments = []
        self.time_segments = []

        # Initialize rewards with reward at initial state
        initial_reward = 0
        self.reward_segments = []

        # Return observation as (ODE copy, state)
        observation = (deepcopy(self.current_ode), self.current_state.clone())
        info = {}

        return observation, info

    def step(self, action: Tuple[ODE, float]) -> Tuple[Tuple[ODE, torch.Tensor], float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: (new_ode, time_to_apply)

        Returns:
            observation: (ODE copy, new_state)
            reward: Sum of rewards over trajectory sampled at reward_dt intervals
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        new_ode, time_delta = action

        # Update current ODE
        self.current_ode = new_ode

        # Find grid points in the interval [current_time, current_time + time_delta]
        end_time = self.current_time + time_delta
        mask = (self.reward_grid >= self.current_time) & (self.reward_grid < end_time)
        grid_points = self.reward_grid[mask]

        # Convert to relative times (starting from 0)
        t_grid = grid_points - self.current_time

        # Always include end_time for proper state update
        t = torch.cat([t_grid, torch.tensor([time_delta])])

        # Remove duplicates and sort
        t = torch.unique(t, sorted=True)

        # Simulate using torchdiffeq
        traj = odeint(self.current_ode, self.current_state, t, method="rk4")

        # Update state to final point (at end_time)
        self.current_state = traj[-1]

        # Update current time
        self.current_time = end_time

        # Extract states at grid points only (exclude end_time if it's not on grid)
        n_grid_points = len(grid_points)
        grid_traj = traj[:n_grid_points]

        # Append trajectory segment (grid points only)
        self.trajectory_segments.append(grid_traj)

        # Append time segment (absolute times)
        self.time_segments.append(grid_points)

        # Compute rewards for grid points only
        rewards = torch.tensor([self.reward_fn(state) for state in grid_traj])

        # Append reward segment
        self.reward_segments.append(rewards)

        # Return sum of rewards
        reward = rewards.sum().item()

        # Return observation as (ODE copy, state)
        observation = (deepcopy(self.current_ode), self.current_state.clone())

        # For now, episodes never terminate
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def get_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get full trajectory by concatenating all segments.

        Returns:
            times: Concatenated time points
            states: Concatenated state trajectory
            rewards: Concatenated rewards
        """
        times = torch.cat(self.time_segments, dim=0)
        states = torch.cat(self.trajectory_segments, dim=0)
        rewards = torch.cat(self.reward_segments, dim=0)
        return times, states, rewards

    def close(self):
        """Clean up environment resources."""
        pass
