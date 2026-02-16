import torch
from typing import Callable
from ..base import ODE


class Antithetic(ODE):
    """Antithetic integral feedback motif with parameter control.

    State Variables:
        Z1: Controller species 1 (integrator)
        Z2: Controller species 2 (anti-integrator)
        B:  Regulated species (output)

    Equations:
        dZ1/dt = mu - (u[0] * eta) * Z1 * Z2
        dZ2/dt = theta * B - (u[0] * eta) * Z1 * Z2
        dB/dt  = Z1 - (u[1] * gamma) * B

    Fixed Parameters (4 total):
        mu:    Production rate of Z1 (default: 1.0)
        eta:   Annihilation rate of Z1*Z2 (default: 1.0)
        theta: Sensing gain of B (default: 1.0)
        gamma: Degradation rate of B (default: 1.0)

    Control Inputs (2 total, all default to 1.0):
        u[0]: Multiplier for eta (annihilation rate)
        u[1]: Multiplier for gamma (degradation rate)
        When all u[i] = 1.0, behavior matches the base system.

    Steady State (with u=[1,1]):
        B_ss = mu / theta
        At steady state: dZ2/dt = 0 => theta * B = eta * Z1 * Z2
        And dZ1/dt = 0 => mu = eta * Z1 * Z2
        So theta * B_ss = mu => B_ss = mu / theta
    """

    name = "Antithetic Integral Feedback"
    variable_names = ["Z1", "Z2", "B"]
    fixed_param_names = ["mu", "eta", "theta", "gamma"]

    def __init__(
        self,
        fixed_params: torch.Tensor | None = None,
        input_signal: Callable[[torch.Tensor], torch.Tensor] | float | None = None,
    ):
        """Initialize antithetic motif ODE.

        Args:
            fixed_params: [mu, eta, theta, gamma]
                         Uses defaults if not provided.
            input_signal: Unused, kept for interface consistency.
        """
        if fixed_params is None:
            fixed_params = torch.tensor([
                1.0,   # mu: production rate of Z1
                1.0,   # eta: annihilation rate
                1.0,   # theta: sensing gain
                1.0,   # gamma: degradation rate of B
            ])

        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the antithetic motif.

        Args:
            t: Time tensor
            x: State tensor [Z1, Z2, B]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [mu, eta, theta, gamma]
            control: Control tensor [u_eta, u_gamma]
                    multiplying eta and gamma. Defaults to all 1.0.

        Returns:
            dx/dt tensor [dZ1/dt, dZ2/dt, dB/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 4, "Expected 4 fixed params"
        assert len(x) == 3, "Expected state [Z1, Z2, B]"

        # Unpack state
        Z1, Z2, B = x[0], x[1], x[2]

        # Get control (default to all 1.0)
        if control is not None:
            assert len(control) >= 2, "Expected 2 control inputs"
            u = control[:2]
        else:
            u = torch.ones(2, dtype=x.dtype, device=x.device)

        # Unpack base parameters
        mu = fixed_params[0]
        eta = u[0] * fixed_params[1]
        theta = fixed_params[2]
        gamma = u[1] * fixed_params[3]

        # Compute derivatives
        annihilation = eta * Z1 * Z2

        dZ1_dt = mu - annihilation
        dZ2_dt = theta * B - annihilation
        dB_dt = Z1 - gamma * B

        return torch.stack([dZ1_dt, dZ2_dt, dB_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"State Variables:\n"
            f"  Z1, Z2: Controller species\n"
            f"  B: Regulated species (output)\n\n"
            f"Equations:\n"
            f"  dZ1/dt = mu - (u[0] * eta) * Z1 * Z2\n"
            f"  dZ2/dt = theta * B - (u[0] * eta) * Z1 * Z2\n"
            f"  dB/dt  = Z1 - (u[1] * gamma) * B\n\n"
            f"Base Parameters:\n"
            f"  mu    = {params[0]:.2f}  (production rate of Z1)\n"
            f"  eta   = {params[1]:.2f}  (annihilation rate)\n"
            f"  theta = {params[2]:.2f}  (sensing gain)\n"
            f"  gamma = {params[3]:.2f}  (degradation rate of B)\n\n"
            f"Steady State: B_ss = mu / theta = {params[0] / params[2]:.4f}\n\n"
            f"Control Inputs (2 total, all default to 1.0):\n"
            f"  u[0]: multiplier for eta (annihilation)\n"
            f"  u[1]: multiplier for gamma (degradation)\n"
            f"  When u=[1,1], behavior matches base system"
        )
