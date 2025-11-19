import torch
import pytest
from rpasim.ode.classic_control import HIVTreatment


def test_hiv_initialization():
    """Test that HIVTreatment initializes with correct parameters."""
    ode = HIVTreatment()

    assert ode.differentiable_params is None
    assert ode.fixed_params is not None
    assert len(ode.fixed_params) == 13

    # Check parameter values [λ, d, β, a, p1, p2, c1, c2, b1, b2, q, h, η]
    expected = torch.tensor([
        1.0, 0.1, 1.0, 0.2, 1.0, 1.0, 0.03, 0.06, 0.1, 0.01, 0.5, 0.1, 0.9799
    ])
    assert torch.allclose(ode.fixed_params, expected)


def test_hiv_variable_names():
    """Test that variable names are correctly defined."""
    ode = HIVTreatment()

    assert len(ode.variable_names) == 5
    assert ode.variable_names == [
        "healthy_CD4", "infected_CD4", "CTL_precursor", "CTL_indep", "CTL_dep"
    ]


def test_hiv_param_names():
    """Test that parameter names are correctly defined."""
    ode = HIVTreatment()

    assert len(ode.fixed_param_names) == 13
    expected_names = [
        "lambda", "d", "beta", "a", "p1", "p2",
        "c1", "c2", "b1", "b2", "q", "h", "eta"
    ]
    assert ode.fixed_param_names == expected_names


def test_hiv_healthy_steady_state():
    """Test at the healthy steady state (recovery fixed point x^B)."""
    ode = HIVTreatment()

    # Extract parameters
    lambda_p, d, beta, a, p1, p2, c1, c2, b1, b2, q, h, eta = ode.fixed_params

    # Compute steady state x^B using equations (6.2a, 6.2b)
    # x2^B = [c2(λ - dq) - b2*β - sqrt([c2(λ - dq) - b2*β]^2 - 4*β*c2*q*d*b2)] / (2*β*c2*q)
    term1 = c2 * (lambda_p - d * q) - b2 * beta
    discriminant = term1**2 - 4 * beta * c2 * q * d * b2
    x2_B = (term1 - torch.sqrt(discriminant)) / (2 * beta * c2 * q)

    # x1^B = λ / (d + β*x2^B)
    x1_B = lambda_p / (d + beta * x2_B)

    # x5^B = [x2^B * c2(βq - a) + b2*β] / (c2*p2*x2^B)
    x5_B = (x2_B * c2 * (beta * q - a) + b2 * beta) / (c2 * p2 * x2_B)

    # x3^B = h*x5^B / (c2*q*x2^B)
    x3_B = (h * x5_B) / (c2 * q * x2_B)

    # x4^B = 0
    x4_B = torch.tensor(0.0)

    x_fixed = torch.tensor([x1_B, x2_B, x3_B, x4_B, x5_B])
    t = torch.tensor(0.0)

    # At steady state, derivatives should be zero
    dx_dt = ode(t, x_fixed)

    assert len(dx_dt) == 5
    assert torch.allclose(dx_dt, torch.zeros(5), atol=1e-4)
