import torch
from ..base import ODE


class FlightControl(ODE):
    """F-8 Crusader aircraft flight control system with nonlinear dynamics.

    From Brunton et al., "Sparse identification of nonlinear dynamics with
    control (SINDy-MPC)", Proc. R. Soc. A 474: 20180335 (2018), Section 5.

    Aircraft dynamics at 30,000 ft (9000 m) and Mach = 0.85.

    Equations (from paper Eq. 5.1, uncontrolled u=0):
        dx1/dt = -0.877*x1 + x3 - 0.088*x1*x3 + 0.47*x1^2 - 0.019*x2^2 - x1^2*x3 + 3.846*x1^3
        dx2/dt = x3
        dx3/dt = -4.208*x1 - 0.396*x3 - 0.47*x1^2 - 3.564*x1^3

    Control effects (when u != 0):
        dx1/dt adds: -0.215*u + 0.28*x1^2*u + 0.47*x1*u^2 + 0.63*u^3
        dx3/dt adds: -20.967*u + 6.265*x1^2*u + 46*x1*u^2 + 61.1*u^3

    Parameters (all fixed):
        alpha1 = -0.877, alpha2 = 1.0, alpha3 = -0.088, alpha4 = 0.47
        alpha5 = -0.019, alpha6 = -1.0, alpha7 = 3.846
        beta1 = 1.0
        gamma1 = -4.208, gamma2 = -0.396, gamma3 = -0.47, gamma4 = -3.564

    Control objective (from paper Eq. 5.2):
        - Track reference trajectory: r(t) = 0.4*(-0.5/(1+exp(-t/0.1-0.8)) + 1/(1+exp(t/0.1-3)) - 0.4)
        - Cost function: Q = 25 (tracking x1), R = 0.05
        - Control limits: u ∈ [-0.3, 0.5] (tail deflection angle)
        - State limit on x1: [-0.2, 0.4] rad (angle of attack constraint)
        - Time horizon: 13 seconds
        - System timestep: Δt = 0.001, Model timestep: Δt^M = 0.01

    State variables:
        x1: angle of attack (rad)
        x2: pitch angle (rad)
        x3: pitch rate (rad/s)

    Control:
        u: tail deflection angle (rad)
    """

    name = "Flight Control System"
    variable_names = ["x1", "x2", "x3"]
    fixed_param_names = [
        "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "alpha7",
        "beta1",
        "gamma1", "gamma2", "gamma3", "gamma4"
    ]

    def __init__(self):
        """Initialize flight control ODE with fixed parameters.

        Fixed parameters: [alpha1, ..., alpha7, beta1, gamma1, ..., gamma4]
        """
        # Fixed parameters: [alpha1-7, beta1, gamma1-4]
        fixed_params = torch.tensor([
            -0.877, 1.0, -0.088, 0.47, -0.019, -1.0, 3.846,  # alpha1-7
            1.0,  # beta1
            -4.208, -0.396, -0.47, -3.564  # gamma1-4
        ])
        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the flight control system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [x1, x2, x3]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [alpha1-7, beta1, gamma1-4]

        Returns:
            dx/dt tensor [dx1/dt, dx2/dt, dx3/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 12, "Expected 12 fixed params"
        assert len(x) == 3, "Expected state [x1, x2, x3]"

        # Unpack state
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # Unpack parameters
        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7 = fixed_params[:7]
        beta1 = fixed_params[7]
        gamma1, gamma2, gamma3, gamma4 = fixed_params[8:]

        # Compute derivatives (uncontrolled system, u=0)
        dx1_dt = (alpha1 * x1 + alpha2 * x3 + alpha3 * x1 * x3 +
                  alpha4 * x1**2 + alpha5 * x2**2 + alpha6 * x1**2 * x3 +
                  alpha7 * x1**3)
        dx2_dt = beta1 * x3
        dx3_dt = gamma1 * x1 + gamma2 * x3 + gamma3 * x1**2 + gamma4 * x1**3

        return torch.stack([dx1_dt, dx2_dt, dx3_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params
        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7 = params[:7]
        beta1 = params[7]
        gamma1, gamma2, gamma3, gamma4 = params[8:]

        return (
            f"{self.name}\n\n"
            f"Equations (uncontrolled, u=0):\n"
            f"  dx1/dt = {alpha1:.2f}*x1 + {alpha2:.2f}*x3 + {alpha3:.2f}*x1*x3 + "
            f"{alpha4:.2f}*x1² + {alpha5:.2f}*x2² + {alpha6:.2f}*x1²*x3 + {alpha7:.2f}*x1³\n"
            f"  dx2/dt = {beta1:.2f}*x3\n"
            f"  dx3/dt = {gamma1:.2f}*x1 + {gamma2:.2f}*x3 + {gamma3:.2f}*x1² + {gamma4:.2f}*x1³\n"
            f"Control limits: u ∈ [-0.3, 0.5], State limit: x1 ∈ [-0.2, 0.4]"
        )
