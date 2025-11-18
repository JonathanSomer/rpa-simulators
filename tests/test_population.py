import torch
import pytest
from rpasim.ode.classic_control import PopulationDynamics


def test_population_dynamics_initialization():
    """Test that PopulationDynamics initializes with correct parameters."""
    ode = PopulationDynamics()

    assert ode.differentiable_params is None
    assert ode.fixed_params is not None
    assert len(ode.fixed_params) == 4

    # Check parameter values [a, b, c, d]
    expected = torch.tensor([0.5, 0.025, 0.5, 0.005])
    assert torch.allclose(ode.fixed_params, expected)


def test_population_dynamics_variable_names():
    """Test that variable names are correctly defined."""
    ode = PopulationDynamics()

    assert len(ode.variable_names) == 2
    assert ode.variable_names == ["prey", "predator"]


def test_population_dynamics_param_names():
    """Test that parameter names are correctly defined."""
    ode = PopulationDynamics()

    assert len(ode.fixed_param_names) == 4
    assert ode.fixed_param_names == ["a", "b", "c", "d"]


def test_population_dynamics_forward():
    """Test forward computation of population dynamics."""
    ode = PopulationDynamics()

    # Test at critical point: x_crit = (c/d, a/b) = (100, 20)
    x_crit = torch.tensor([100.0, 20.0])
    t = torch.tensor(0.0)

    dx_dt = ode(t, x_crit)

    # At critical point, derivatives should be zero
    assert len(dx_dt) == 2
    assert torch.allclose(dx_dt, torch.zeros(2), atol=1e-6)


def test_population_dynamics_away_from_equilibrium():
    """Test dynamics away from equilibrium."""
    ode = PopulationDynamics()

    # Test with different initial conditions
    x = torch.tensor([50.0, 10.0])  # Less prey and predators than critical point
    t = torch.tensor(0.0)

    dx_dt = ode(t, x)

    # dx1/dt = 0.5*50 - 0.025*50*10 = 25 - 12.5 = 12.5 (prey increasing)
    # dx2/dt = -0.5*10 + 0.005*50*10 = -5 + 2.5 = -2.5 (predators decreasing)
    expected = torch.tensor([12.5, -2.5])

    assert torch.allclose(dx_dt, expected, atol=1e-6)
