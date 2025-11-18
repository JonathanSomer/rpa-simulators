import torch
import pytest
from rpasim.ode.classic_control import Lorenz


def test_lorenz_initialization():
    """Test that Lorenz initializes with correct parameters."""
    ode = Lorenz()

    assert ode.differentiable_params is None
    assert ode.fixed_params is not None
    assert len(ode.fixed_params) == 3

    # Check parameter values [sigma, beta, rho]
    expected = torch.tensor([10.0, 8.0/3.0, 28.0])
    assert torch.allclose(ode.fixed_params, expected)


def test_lorenz_variable_names():
    """Test that variable names are correctly defined."""
    ode = Lorenz()

    assert len(ode.variable_names) == 3
    assert ode.variable_names == ["x1", "x2", "x3"]


def test_lorenz_param_names():
    """Test that parameter names are correctly defined."""
    ode = Lorenz()

    assert len(ode.fixed_param_names) == 3
    assert ode.fixed_param_names == ["sigma", "beta", "rho"]


def test_lorenz_fixed_point():
    """Test at unstable fixed point where derivatives should be zero."""
    ode = Lorenz()

    # Fixed point: (sqrt(72), sqrt(72), 27)
    sqrt_72 = torch.sqrt(torch.tensor(72.0))
    x_fixed = torch.tensor([sqrt_72, sqrt_72, 27.0])
    t = torch.tensor(0.0)

    dx_dt = ode(t, x_fixed)

    # At fixed point, all derivatives should be zero
    assert len(dx_dt) == 3
    assert torch.allclose(dx_dt, torch.zeros(3), atol=1e-5)


def test_lorenz_chaotic_dynamics():
    """Test dynamics away from fixed point."""
    ode = Lorenz()

    # Test with state near origin
    x = torch.tensor([1.0, 1.0, 1.0])
    t = torch.tensor(0.0)

    dx_dt = ode(t, x)

    # With sigma=10, beta=8/3, rho=28:
    # dx1/dt = 10*(1 - 1) = 0
    # dx2/dt = 1*(28 - 1) - 1 = 26
    # dx3/dt = 1*1 - (8/3)*1 = -5/3
    expected = torch.tensor([0.0, 26.0, -5.0/3.0])

    assert torch.allclose(dx_dt, expected, atol=1e-6)
