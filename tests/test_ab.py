import torch
import pytest
from torchdiffeq import odeint
from rpasim.ode import AB


def test_ab_parameter_names():
    """Test that AB ODE has correct parameter names defined."""
    ab_ode = AB()

    assert ab_ode.variable_names == ["A", "B"]
    assert ab_ode.differentiable_param_names == ["alpha1", "alpha2", "alpha3"]
    assert ab_ode.fixed_param_names == ["beta1", "beta2"]


def test_ab_steady_state():
    """Test that AB ODE reaches steady state where x1 = 1/abs(alpha3)"""
    # Parameters
    alpha1 = 1.0
    alpha2 = 0.0
    beta1 = 10.0
    beta2 = 1.0

    # Test with multiple values of alpha3
    alpha3_values = torch.linspace(-1/10, -1/2, 10)  # -0.1 to -0.5

    for alpha3 in alpha3_values:
        # Create ODE with varying alpha3
        differentiable_params = torch.tensor([alpha1, alpha2, alpha3.item()])
        fixed_params = torch.tensor([beta1, beta2])
        ab_ode = AB(differentiable_params=differentiable_params, fixed_params=fixed_params)

        # Initial state
        x0 = torch.tensor([1.0, 0.5])

        # Simulate for long time to reach steady state
        T = 100.0
        t = torch.linspace(0, T, 1000)
        trajectory = odeint(ab_ode, x0, t)

        # Get final state (steady state)
        x1_steady = trajectory[-1, -1].item()  # Last variable (B)

        # Expected steady state: x1 = 1/abs(alpha3)
        expected_x1 = 1.0 / abs(alpha3.item())

        # Assert with relative tolerance
        assert x1_steady == pytest.approx(expected_x1, rel=1e-2), \
            f"For alpha3={alpha3.item():.4f}, expected x1={expected_x1:.4f}, got {x1_steady:.4f}"
