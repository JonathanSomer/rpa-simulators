import torch
from ..base import ODE


class PopulationDynamics(ODE):
    """Lotka-Volterra predator-prey population dynamics.

    From Brunton et al., "Sparse identification of nonlinear dynamics with
    control (SINDy-MPC)", Proc. R. Soc. A 474: 20180335 (2018), Section 3.

    Two-dimensional, weakly nonlinear system describing interaction between
    competing populations. May represent biological species, competition in
    stock markets, or spread of infectious diseases.

    Equations (from paper Eq. 3.1, uncontrolled u=0):
        dx1/dt = a*x1 - b*x1*x2  (prey population)
        dx2/dt = -c*x2 + d*x1*x2  (predator population)

    Control effects (when u != 0):
        dx2/dt = -c*x2 + d*x1*x2 + u  (control affects predator only)

    Parameters (all fixed):
        a = 0.5   (prey growth rate)
        b = 0.025 (effect of predation on prey population)
        c = 0.5   (predator death rate)
        d = 0.005 (predator growth based on prey population size)

    Fixed point:
        Critical point x^crit = (c/d, a/b)^T = (100, 20)^T
        At this point, population sizes are in balance.

    System behavior:
        - Unforced system exhibits limit cycle behavior (predator lags prey)
        - Control objective: stabilize the critical point

    Control objective:
        - Stabilize critical point (100, 20)^T
        - Cost function: Q = I₂ (2×2 identity), R_u = R_Δu = 0.5
        - Control limits: u ∈ [-20, 20]
        - Constraint: x2 ≥ 10 (minimum predator population for recovery)

    State variables:
        x1: prey population size
        x2: predator population size

    Control:
        u: Control input affecting predator population only
    """

    name = "Population Dynamics (Lotka-Volterra)"
    variable_names = ["prey", "predator"]
    fixed_param_names = ["a", "b", "c", "d"]

    def __init__(self):
        """Initialize population dynamics ODE with fixed parameters.

        Fixed parameters: [a, b, c, d]
        """
        # Fixed parameters: [a, b, c, d]
        fixed_params = torch.tensor([0.5, 0.025, 0.5, 0.005])
        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the population dynamics system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [prey, predator]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [a, b, c, d]
            control: Control input tensor (affects predator only)
                    Can be scalar or tensor of shape (1,) for single control input

        Returns:
            dx/dt tensor [dprey/dt, dpredator/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 4, "Expected 4 fixed params [a, b, c, d]"
        assert len(x) == 2, "Expected state [prey, predator]"

        # Unpack state
        x1 = x[0]  # prey
        x2 = x[1]  # predator

        # Unpack parameters
        a, b, c, d = fixed_params

        # Extract control input (default to 0 if not provided)
        if control is not None:
            u = control[0] if control.dim() > 0 else control
        else:
            u = torch.tensor(0.0)

        # Compute derivatives
        dx1_dt = a * x1 - b * x1 * x2
        dx2_dt = -c * x2 + d * x1 * x2 + u  # Control affects predator

        return torch.stack([dx1_dt, dx2_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        a, b, c, d = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"Equations:\n"
            f"  dprey/dt = {a:.2f}*prey - {b:.2f}*prey*predator\n"
            f"  dpredator/dt = -{c:.2f}*predator + {d:.2f}*prey*predator\n"
            f"Parameters:\n"
            f"  a = {a:.2f}, b = {b:.2f}, c = {c:.2f}, d = {d:.2f}"
        )
