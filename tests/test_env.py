import torch
import pytest
from rpasim.ode import ODE
from rpasim.env import DifferentiableEnv


class ConstantODE(ODE):
    """Simple ODE with constant derivative: dx/dt = c.

    This means x(t) = x0 + c*t
    """

    variable_names = ["x"]

    def __init__(self, c: float):
        """Initialize with constant derivative c."""
        super().__init__()
        self.c = c

    def forward(self, t, x, differentiable_params=None, fixed_params=None):
        """Return constant derivative."""
        return torch.tensor([self.c])

    def __str__(self) -> str:
        """Return string representation."""
        return f"ConstantODE(c={self.c})"


def test_state_limit_truncation():
    """Test that state limits trigger truncation with correct reward."""
    # Setup with coarse grid and fast dynamics for exact computation
    # dx/dt = 150, x0 = 0, x_limit = 100
    # So x(t) = 150*t, crosses limit at t = 100/150 ≈ 0.67
    c = 150.0
    x0 = torch.tensor([0.0])
    x_limit = 100.0
    T = 10.0  # Time horizon
    n_reward_steps = 10  # Coarse grid: dt = 1.0
    # Grid points: 0, 1, 2, 3, ..., 10

    ode = ConstantODE(c)

    # Reward function: simple negative x
    def reward_fn(state, t):
        return -state[0]

    # Create env with state limit
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=n_reward_steps,
        state_limits=(0.0, x_limit),  # Upper limit at 100
    )

    # Reset env
    obs, info = env.reset()

    # Take one long step that should trigger truncation
    action = (ode, T)  # Try to go full time horizon
    obs, reward, terminated, truncated, info = env.step(action)

    # Verify truncation occurred
    assert truncated, "Environment should be truncated when state limit is exceeded"
    assert not terminated, "Environment should not terminate on truncation"

    # With this setup, we expect:
    # Grid point 0 (t=0): x=0
    # Grid point 1 (t=1): x=150 (exceeds limit, violation detected here)
    assert len(env.trajectory_segments[0]) == 2, \
        f"Should have sampled 2 grid points, got {len(env.trajectory_segments[0])}"

    # Verify the trajectory values
    traj = env.trajectory_segments[0]
    assert abs(traj[0, 0].item() - 0.0) < 0.01, f"First point should be 0, got {traj[0, 0].item()}"
    assert abs(traj[1, 0].item() - 150.0) < 0.01, f"Second point should be 150, got {traj[1, 0].item()}"

    # Exact computation of expected reward:
    # Grid point 0: x=0, reward = -0 = 0
    # Grid point 1: x=150, reward = -150
    reward_from_grid = 0.0 + (-150.0)

    # Total grid points from 0 to T: 0, 1, 2, ..., 10 (11 points)
    # Sampled: 2 (points 0 and 1)
    # Remaining: 2, 3, 4, ..., 10 (9 points)
    n_remaining = 9

    # Final state at truncation: 150
    final_state = env.current_state[0].item()
    assert abs(final_state - 150.0) < 0.01, f"Final state should be 150, got {final_state}"

    # Penalty: reward_fn(final_state) * n_remaining = -150 * 9 = -1350
    penalty = -final_state * n_remaining

    expected_reward = reward_from_grid + penalty  # -150 + (-1350) = -1500

    # Compare
    assert abs(reward.item() - expected_reward) < 0.01, \
        f"Expected reward {expected_reward:.4f}, got {reward.item():.4f}"


def test_no_truncation_without_limits():
    """Test that env works normally without state limits."""
    c = 2.0
    x0 = torch.tensor([0.0])
    T = 10.0
    n_reward_steps = 50

    ode = ConstantODE(c)

    def reward_fn(state, t):
        return -state[0]

    # Create env WITHOUT state limits
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=n_reward_steps,
    )

    obs, info = env.reset()
    action = (ode, T)
    obs, reward, terminated, truncated, info = env.step(action)

    # Should terminate (reach time horizon) but not truncate
    assert terminated, "Should terminate when reaching time horizon"
    assert not truncated, "Should not truncate without state limits"


def test_trajectory_consistency_across_grid_widths():
    """Test that different grid widths produce same trajectory at shared points."""
    from rpasim.ode import AB

    # Setup AB ODE
    differentiable_params = torch.tensor([1.0, 0.0, -0.2])
    fixed_params = torch.tensor([10.0, 1.0])
    ab_ode = AB(differentiable_params, fixed_params)

    x0 = torch.tensor([1.0, 0.5])
    T = 10.0

    def reward_fn(state, t):
        return -torch.norm(state)

    # Create envs with different grid widths
    # Coarse: 20 steps (dt=0.5), grid at 0, 0.5, 1, ..., 10
    env_coarse = DifferentiableEnv(
        initial_ode=ab_ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=20,
    )

    # Fine: 100 steps (dt=0.1), grid at 0, 0.1, 0.2, ..., 10
    env_fine = DifferentiableEnv(
        initial_ode=ab_ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=100,
    )

    # Run both
    env_coarse.reset()
    env_coarse.step((ab_ode, T))

    env_fine.reset()
    env_fine.step((ab_ode, T))

    # Get trajectories
    times_coarse, states_coarse, _ = env_coarse.get_trajectory()
    times_fine, states_fine, _ = env_fine.get_trajectory()

    # Find indices in fine grid that match coarse grid times
    # For each coarse time, find the closest fine time
    matches = []
    for t_coarse in times_coarse:
        time_diffs = torch.abs(times_fine - t_coarse)
        idx_fine = torch.argmin(time_diffs)
        if time_diffs[idx_fine] < 0.01:  # Within tolerance
            matches.append(idx_fine.item())

    # Extract matched states from fine grid
    states_fine_matched = states_fine[matches]

    # Compare all matched states using tensor operations
    state_diffs = torch.norm(states_coarse - states_fine_matched, dim=1)
    max_diff = state_diffs.max().item()

    assert max_diff < 0.05, \
        f"States at shared grid points differ by up to {max_diff:.4f} (should be < 0.05)"


def test_trajectory_consistency_single_vs_multi_step():
    """Test that running in one step vs multiple steps gives similar results."""
    from rpasim.ode import AB

    # Setup AB ODE
    differentiable_params = torch.tensor([1.0, 0.0, -0.2])
    fixed_params = torch.tensor([10.0, 1.0])
    ab_ode = AB(differentiable_params, fixed_params)

    x0 = torch.tensor([1.0, 0.5])
    T = 10.0
    n_reward_steps = 100

    def reward_fn(state, t):
        return -torch.norm(state)

    # Single step environment
    env_single = DifferentiableEnv(
        initial_ode=ab_ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=n_reward_steps,
    )

    # Multi step environment
    env_multi = DifferentiableEnv(
        initial_ode=ab_ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=n_reward_steps,
    )

    # Run single step
    env_single.reset()
    _, reward_single, _, _, _ = env_single.step((ab_ode, T))

    # Run multiple steps
    env_multi.reset()
    n_steps = 10
    step_size = T / n_steps
    total_reward_multi = torch.tensor(0.0)
    for _ in range(n_steps):
        _, reward, terminated, truncated, _ = env_multi.step((ab_ode, step_size))
        total_reward_multi += reward
        if terminated or truncated:
            break

    # Get trajectories
    times_single, states_single, _ = env_single.get_trajectory()
    times_multi, states_multi, _ = env_multi.get_trajectory()

    # All states at shared grid points should be similar
    assert len(times_single) == len(times_multi), \
        f"Should have same number of grid points: single={len(times_single)}, multi={len(times_multi)}"

    # Compare all states using tensor operations
    state_diffs = torch.norm(states_single - states_multi, dim=1)
    max_diff = state_diffs.max().item()

    assert max_diff < 0.1, \
        f"States differ by up to {max_diff:.4f} (should be < 0.1)"

    # Total rewards should be similar
    reward_diff = abs(reward_single.item() - total_reward_multi.item())
    assert reward_diff < 1.0, \
        f"Total rewards differ: single={reward_single.item():.2f}, multi={total_reward_multi.item():.2f}, diff={reward_diff}"


def test_time_dependent_reward():
    """Test that time parameter is correctly passed to reward function."""
    # Track which times the reward function was called with
    times_called = []

    def reward_fn(state, t):
        times_called.append(t.item())
        # Time-varying reference: r(t) = t
        reference = t
        return -(state[0] - reference) ** 2

    # Setup: dx/dt = 0, x0 = 5.0, so x(t) = 5.0 for all t
    # Reference: r(t) = t
    # At each time t, tracking error = x(t) - r(t) = 5.0 - t
    # Reward = -(error)^2 = -(5.0 - t)^2
    c = 0.0
    x0 = torch.tensor([5.0])
    T = 10.0
    n_reward_steps = 10  # Grid: 0, 1, 2, ..., 10

    ode = ConstantODE(c)
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=reward_fn,
        initial_state=x0,
        time_horizon=T,
        n_reward_steps=n_reward_steps,
    )

    # Run environment
    env.reset()
    env.step((ode, T))

    # Verify times match grid points (0, 1, 2, ..., 9)
    # Note: grid points are sampled at [current_time, end_time), excluding endpoint
    expected_times = [float(i) for i in range(10)]
    assert len(times_called) == len(expected_times), \
        f"Expected {len(expected_times)} time points, got {len(times_called)}"

    for i, (actual, expected) in enumerate(zip(times_called, expected_times)):
        assert abs(actual - expected) < 0.01, \
            f"Time point {i}: expected {expected}, got {actual}"

    # Verify reward computation with time-dependent reference
    # At t=5: x=5.0, r=5.0, error=0, reward=0
    times, states, rewards = env.get_trajectory()

    # Manually compute expected reward at t=5 (6th point, index 5)
    t_test = 5.0
    x_test = 5.0  # x(t) = 5.0 (constant)
    r_test = t_test  # r(5) = 5.0
    expected_reward_at_5 = -((x_test - r_test) ** 2)  # -(5.0-5.0)^2 = 0

    actual_reward_at_5 = rewards[5].item()
    assert abs(actual_reward_at_5 - expected_reward_at_5) < 0.01, \
        f"At t=5: expected reward {expected_reward_at_5:.2f}, got {actual_reward_at_5:.2f}"
