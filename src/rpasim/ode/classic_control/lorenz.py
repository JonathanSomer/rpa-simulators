import torch
from ..base import ODE


class Lorenz(ODE):
    """Chaotic Lorenz system modeling Rayleigh-Bénard convection.

    From Brunton et al., "Sparse identification of nonlinear dynamics with
    control (SINDy-MPC)", Proc. R. Soc. A 474: 20180335 (2018), Section 4.

    Prototypical example of chaos in dynamical systems. Originally proposed
    by Lorenz [65] for atmospheric convection.

    Equations (from paper Eq. 4.1, uncontrolled u=0):
        dx1/dt = σ(x2 - x1)
        dx2/dt = x1(ρ - x3) - x2
        dx3/dt = x1*x2 - β*x3

    Control effects (when u != 0):
        dx1/dt = σ(x2 - x1) + u  (control affects only first state)

    Parameters (all fixed):
        σ = 10      (Prandtl number)
        β = 8/3     (geometric factor)
        ρ = 28      (Rayleigh number)

    Fixed points:
        Two weakly unstable fixed points: (±√72, ±√72, 27)ᵀ ≈ (±8.49, ±8.49, 27)ᵀ
        Trajectories typically oscillate alternately around these points.

    Control objective:
        - Stabilize one of the fixed points
        - Cost function: Q = I₃ (3×3 identity), R_u = R_Δu = 0.001
        - Control limits: u ∈ [-50, 50]

    State variables:
        x1, x2, x3: State components (dimensionless)

    Control:
        u: Control input affecting x1 only
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
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the Lorenz system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [x1, x2, x3]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [sigma, beta, rho]
            control: Control input tensor (affects x1 only)
                    Can be scalar or tensor of shape (1,) for single control input

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

        # Extract control input (default to 0 if not provided)
        if control is not None:
            u = control[0] if control.dim() > 0 else control
        else:
            u = torch.tensor(0.0)

        # Compute derivatives
        dx1_dt = sigma * (x2 - x1) + u  # Control affects x1
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
