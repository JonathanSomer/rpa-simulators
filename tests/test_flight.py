import torch
import pytest
from rpasim.ode.classic_control import FlightControl


def test_flight_initialization():
    """Test that FlightControl initializes with correct parameters."""
    ode = FlightControl()

    assert ode.differentiable_params is None
    assert ode.fixed_params is not None
    assert len(ode.fixed_params) == 12

    # Check parameter values [alpha1-7, beta1, gamma1-4]
    expected = torch.tensor([
        -0.877, 1.0, -0.088, 0.47, -0.019, -1.0, 3.846,  # alpha1-7
        1.0,  # beta1
        -4.208, -0.396, -0.47, -3.564  # gamma1-4
    ])
    assert torch.allclose(ode.fixed_params, expected)


def test_flight_variable_names():
    """Test that variable names are correctly defined."""
    ode = FlightControl()

    assert len(ode.variable_names) == 3
    assert ode.variable_names == ["x1", "x2", "x3"]


def test_flight_param_names():
    """Test that parameter names are correctly defined."""
    ode = FlightControl()

    assert len(ode.fixed_param_names) == 12
    expected_names = [
        "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "alpha7",
        "beta1",
        "gamma1", "gamma2", "gamma3", "gamma4"
    ]
    assert ode.fixed_param_names == expected_names


def test_flight_forward():
    """Test forward computation of flight control system."""
    ode = FlightControl()

    # Test at origin
    x = torch.tensor([0.0, 0.0, 0.0])
    t = torch.tensor(0.0)

    dx_dt = ode(t, x)

    # At origin, all derivatives should be zero
    assert len(dx_dt) == 3
    assert torch.allclose(dx_dt, torch.zeros(3), atol=1e-6)


def test_flight_nonlinear_dynamics():
    """Test dynamics at a nonzero state."""
    ode = FlightControl()

    # Test with specific state
    x = torch.tensor([0.1, 0.2, 0.3])
    t = torch.tensor(0.0)

    dx_dt = ode(t, x)

    # Manually compute expected values
    x1, x2, x3 = 0.1, 0.2, 0.3

    # dx1/dt = -0.877*0.1 + 1.0*0.3 - 0.088*0.1*0.3 + 0.47*0.01 - 0.019*0.04 - 1.0*0.01*0.3 + 3.846*0.001
    dx1_expected = (-0.877 * x1 + 1.0 * x3 - 0.088 * x1 * x3 +
                    0.47 * x1**2 - 0.019 * x2**2 - 1.0 * x1**2 * x3 +
                    3.846 * x1**3)

    # dx2/dt = 1.0*0.3
    dx2_expected = 1.0 * x3

    # dx3/dt = -4.208*0.1 - 0.396*0.3 - 0.47*0.01 - 3.564*0.001
    dx3_expected = -4.208 * x1 - 0.396 * x3 - 0.47 * x1**2 - 3.564 * x1**3

    expected = torch.tensor([dx1_expected, dx2_expected, dx3_expected])

    assert len(dx_dt) == 3
    assert torch.allclose(dx_dt, expected, atol=1e-6)
