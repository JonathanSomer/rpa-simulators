import torch
from typing import Callable, Tuple, Dict, Optional, Union, List
from torchdiffeq import odeint, odeint_event
from rpasim.ode import ODE


def _process_state_limits(
    state_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
    n_vars: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Process state limits into normalized format.

    Args:
        state_limits: Single tuple (lower, upper) or list of tuples per variable
        n_vars: Number of state variables

    Returns:
        Tuple of (lower_bounds, upper_bounds) tensors, or None
    """
    if state_limits is None:
        return None

    if isinstance(state_limits, tuple):
        # Single tuple: apply to all variables
        lower_bounds = torch.full((n_vars,), float(state_limits[0]))
        upper_bounds = torch.full((n_vars,), float(state_limits[1]))
    else:
        # List of tuples: one per variable
        assert len(state_limits) == n_vars, f"Expected {n_vars} limit tuples, got {len(state_limits)}"
        lower_bounds = torch.tensor([lim[0] for lim in state_limits], dtype=torch.float)
        upper_bounds = torch.tensor([lim[1] for lim in state_limits], dtype=torch.float)

    return (lower_bounds, upper_bounds)


def _make_event_fn(state_limits: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    """Create an event function for state limit violations.

    Args:
        state_limits: Tuple of (lower_bounds, upper_bounds)

    Returns:
        Event function that returns 0 when any state violates limits,
        or None if no limits are set
    """
    if state_limits is None:
        return None

    lower_bounds, upper_bounds = state_limits

    def event_fn(t, y):
        # Add small epsilon to bounds to avoid triggering at exact boundary
        epsilon = 1e-9
        # Compute margin from lower and upper bounds for each variable
        margin_lower = y - (lower_bounds - epsilon)
        margin_upper = (upper_bounds + epsilon) - y

        # Get minimum margin across all variables and bounds
        min_margin = torch.min(torch.min(margin_lower), torch.min(margin_upper))

        # Clamp to 0 so event_fn is exactly 0 when violated
        return torch.max(min_margin, torch.tensor(0.0))

    return event_fn


class DifferentiableEnv:
    """Differentiable environment for ODE control.

    Implements the gymnasium Env API without inheritance.

    Action space: (ODE, time) tuples
    Observation space: (ODE, state) tuples
    """

    def __init__(
        self,
        initial_ode: ODE,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor = None,
        time_horizon: float = 10.0,
        n_reward_steps: int = 1000,
        state_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        initial_state_range: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    ):
        """Initialize the differentiable environment.

        Args:
            initial_ode: Initial ODE instance
            reward_fn: Function mapping (state, time) -> reward
            initial_state: Initial state tensor (used if initial_state_range is None)
            time_horizon: Maximum time for the environment
            n_reward_steps: Number of steps for reward computation grid
            state_limits: Optional limits for state variables. Can be:
                - Single tuple (lower, upper) applied to all variables
                - List of tuples, one per variable
            initial_state_range: Optional range for randomizing initial state on each reset. Can be:
                - Single tuple (lower, upper) applied to all state variables
                - List of tuples, one per state variable
                If provided, initial_state will be sampled uniformly from this range on each reset.
        """
        self.initial_ode = initial_ode
        self.reward_fn = reward_fn

        # Handle initial state range for randomization
        self.initial_state_range = initial_state_range
        if initial_state_range is not None:
            # Determine number of state variables from the range
            if isinstance(initial_state_range, tuple):
                # Need to get n_vars from initial_state or infer from ODE
                if initial_state is not None:
                    n_vars = len(initial_state)
                else:
                    raise ValueError("Must provide initial_state to determine state dimension when using single-tuple initial_state_range")
            else:
                # List of tuples - number of variables is clear
                n_vars = len(initial_state_range)

            # Process range into (lower_bounds, upper_bounds)
            self.initial_state_bounds = _process_state_limits(initial_state_range, n_vars)
            # Set a default initial_state (will be overwritten on first reset)
            if initial_state is None:
                lower_bounds, upper_bounds = self.initial_state_bounds
                initial_state = (lower_bounds + upper_bounds) / 2
        else:
            self.initial_state_bounds = None
            if initial_state is None:
                raise ValueError("Must provide either initial_state or initial_state_range")

        self.initial_state = initial_state
        self.time_horizon = time_horizon
        self.n_reward_steps = n_reward_steps
        self.reward_dt = time_horizon / n_reward_steps
        self.state_limits = _process_state_limits(state_limits, len(initial_state))

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

        Args:
            seed: Random seed (currently unused, for future compatibility)
            options: Additional options (currently unused)

        Returns:
            observation: (ODE, state)
            info: Empty dict
        """
        # Reset to initial state (no deepcopy for gradient flow)
        self.current_ode = self.initial_ode

        # Sample random initial state if range is provided, otherwise use fixed initial state
        if self.initial_state_bounds is not None:
            lower_bounds, upper_bounds = self.initial_state_bounds
            # Sample uniformly within bounds
            random_factors = torch.rand_like(lower_bounds)
            self.current_state = lower_bounds + random_factors * (upper_bounds - lower_bounds)
        else:
            self.current_state = self.initial_state.clone()

        self.current_time = 0.0

        # Initialize empty trajectory
        self.trajectory_segments = []
        self.time_segments = []

        # Initialize rewards with reward at initial state
        initial_reward = 0
        self.reward_segments = []

        # Return observation as (ODE, state)
        observation = (self.current_ode, self.current_state.clone())
        info = {}

        return observation, info

    def step(self, action: Tuple[ODE, float]) -> Tuple[Tuple[ODE, torch.Tensor], float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: (new_ode, time_to_apply)

        Returns:
            observation: (ODE, new_state)
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
        event_fn = _make_event_fn(self.state_limits)

        try:
            # Try regular odeint first (most efficient path)
            traj = odeint(self.current_ode, self.current_state, t, method="rk4")

            # Check for state limit violations on the grid
            violation_idx = None
            if event_fn is not None:
                for idx, state in enumerate(traj):
                    if (event_fn(t[idx], state) == 0).any():
                        violation_idx = idx
                        break

            if violation_idx is None:
                # No violation: proceed normally
                self.current_state = traj[-1]
                self.current_time = end_time
                n_grid_points = len(grid_points)
                grid_traj = traj[:n_grid_points]
                truncated = False
                remaining_grid_points = 0
            else:
                # Truncate trajectory at violation point
                traj = traj[: violation_idx + 1]
                n_grid_points = min(violation_idx + 1, len(grid_points))
                grid_traj = traj[:n_grid_points]
                grid_points = grid_points[:n_grid_points]

                # Update state and time to violation point
                self.current_state = traj[-1]
                self.current_time = self.current_time + t[violation_idx].item()

                # Count remaining grid points from violation time to time horizon
                remaining_grid_points = (
                    ((self.reward_grid > self.current_time) & (self.reward_grid <= self.time_horizon)).sum().item()
                )
                truncated = True

        except (RuntimeError, ValueError) as e:
            # ODE solver failed - likely numerical explosion
            if event_fn is None:
                raise ValueError(
                    f"ODE integration failed with error: {e}. "
                    "This is likely due to numerical instability. "
                    "Consider adding state_limits to prevent dynamics from exploding."
                ) from e

            # Use event handling to find exact stopping time
            t_span = torch.tensor([0.0, time_delta])
            event_t, event_y = odeint(
                self.current_ode,
                self.current_state,
                t_span,
                event_fn=event_fn,
                method="rk4",
                atol=1e-6,
                rtol=1e-3,
            )
            actual_end_time = event_t.item()

            # Get trajectory up to stopping time with grid points
            t_eval = t[t <= actual_end_time]
            traj = odeint(self.current_ode, self.current_state, t_eval, method="rk4")

            # Update state and time
            self.current_state = traj[-1]
            self.current_time = self.current_time + actual_end_time

            # Determine which points are on the reward grid
            n_grid_points = (grid_points <= self.current_time).sum().item()
            grid_traj = traj[:n_grid_points]
            grid_points = grid_points[:n_grid_points]

            # Count remaining grid points
            remaining_grid_points = (
                ((self.reward_grid > self.current_time) & (self.reward_grid <= self.time_horizon)).sum().item()
            )
            truncated = True

        # Append trajectory segment (grid points only)
        self.trajectory_segments.append(grid_traj)

        # Append time segment (absolute times)
        self.time_segments.append(grid_points)

        # Compute rewards for grid points only
        rewards = torch.stack([self.reward_fn(state, t) for state, t in zip(grid_traj, grid_points)])

        # If truncated, add penalty for remaining grid points to horizon
        if truncated and remaining_grid_points > 0:
            # Use clamped boundary state for penalty, not the exploded state
            # This gives a more reasonable penalty based on "staying at the limit"
            if self.state_limits is not None:
                lower_bounds, upper_bounds = self.state_limits
                boundary_state = torch.clamp(self.current_state, lower_bounds, upper_bounds)
            else:
                boundary_state = self.current_state

            boundary_reward = self.reward_fn(boundary_state, torch.tensor(self.current_time))
            penalty = boundary_reward * remaining_grid_points
            rewards = torch.cat([rewards, penalty.unsqueeze(0)])

        # Append reward segment
        self.reward_segments.append(rewards)

        # Return sum of rewards (as tensor for gradient computation)
        reward = rewards.sum()

        # Return observation as (ODE, state)
        observation = (self.current_ode, self.current_state.clone())

        # Check if we've reached or passed the time horizon
        terminated = self.current_time >= self.time_horizon
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
