import torch
from ..base import ODE


class Lorenz(ODE):
    """Chaotic Lorenz system ODE.

    Classic chaotic system exhibiting sensitive dependence on initial conditions.

    Equations:
        dx1/dt = sigma*(x2 - x1)
        dx2/dt = x1*(rho - x3) - x2
        dx3/dt = x1*x2 - beta*x3

    Parameters (all fixed):
        sigma = 10      (Prandtl number)
        beta = 8/3      (geometric factor)
        rho = 28        (Rayleigh number)

    Control objective:
        - Control adds to x1: dx1/dt = sigma*(x2 - x1) + u
        - Control limits: u ∈ [-50, 50]
        - Cost function: Q = I (3x3 identity), R = 0.001
        - Time horizon: 10

    State:
        x: [x1, x2, x3]
    """

    name = "Lorenz System (Chaotic)"
    variable_names = ["x1", "x2", "x3"]
    fixed_param_names = ["sigma", "beta", "rho"]

    def __init__(self):
        """Initialize Lorenz system ODE with fixed parameters.

        Fixed parameters: [sigma, beta, rho]
        """
        # Fixed parameters: [sigma, beta, rho]
        fixed_params = torch.tensor([10.0, 8.0/3.0, 28.0])
        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the Lorenz system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [x1, x2, x3]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [sigma, beta, rho]

        Returns:
            dx/dt tensor [dx1/dt, dx2/dt, dx3/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 3, "Expected 3 fixed params [sigma, beta, rho]"
        assert len(x) == 3, "Expected state [x1, x2, x3]"

        # Unpack state
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # Unpack parameters
        sigma, beta, rho = fixed_params

        # Compute derivatives
        dx1_dt = sigma * (x2 - x1)
        dx2_dt = x1 * (rho - x3) - x2
        dx3_dt = x1 * x2 - beta * x3

        return torch.stack([dx1_dt, dx2_dt, dx3_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        sigma, beta, rho = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"Equations:\n"
            f"  dx1/dt = {sigma:.2f}*(x2 - x1)\n"
            f"  dx2/dt = x1*({rho:.2f} - x3) - x2\n"
            f"  dx3/dt = x1*x2 - {beta:.2f}*x3\n"
            f"Parameters:\n"
            f"  sigma = {sigma:.2f}, beta = {beta:.2f}, rho = {rho:.2f}"
        )
