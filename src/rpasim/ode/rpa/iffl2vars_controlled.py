import torch
from typing import Callable
from ..base import ODE


class IFFL2VarsControlled(ODE):
    """2-variable system with parameter control.

    Same dynamics as IFFL2Vars, but the input signal is fixed (uncontrolled)
    and instead the 4 parameters are each controlled by a separate input signal.

    State Variables:
        x: First state variable
        y: Second state variable

    Equations:
        dx/dt = (u[0] * alpha) * Input(t) - (u[1] * delta) * x
        dy/dt = (u[2] * beta) * Input(t) - (u[3] * gamma) * x * y

    Fixed Parameters (4 total, serve as base values):
        alpha: Input effect on x (default: 1.0)
        delta: Degradation rate of x (default: 0.1)
        beta: Input effect on y (default: 1.0)
        gamma: Interaction rate between x and y (default: 1.0)

    External Input Signal (not controllable):
        Input(t): Time-dependent signal (default: constant 1.0)

    Control Inputs (4 total, all default to 1.0):
        u[0]: Multiplier for alpha
        u[1]: Multiplier for delta
        u[2]: Multiplier for beta
        u[3]: Multiplier for gamma
        When all u[i] = 1.0, behavior matches the base system.
    """

    name = "2-Variable System with Parameter Control"
    variable_names = ["x", "y"]
    fixed_param_names = ["alpha", "delta", "beta", "gamma"]

    def __init__(
        self,
        fixed_params: torch.Tensor | None = None,
        input_signal: Callable[[torch.Tensor], torch.Tensor] | float = 1.0,
    ):
        """Initialize 2-variable ODE with parameters.

        Args:
            fixed_params: [alpha, delta, beta, gamma]
                         Uses defaults if not provided.
            input_signal: Time-dependent input signal function Input(t) or constant value.
                         If callable, will be evaluated at time t.
                         If float, treated as constant signal.
                         Defaults to 1.0.
        """
        if fixed_params is None:
            fixed_params = torch.tensor([
                1.0,   # alpha: Input effect on x
                0.1,   # delta: Degradation rate of x
                1.0,   # beta: Input effect on y
                1.0,   # gamma: Interaction rate between x and y
            ])

        super().__init__(differentiable_params=None, fixed_params=fixed_params)

        if callable(input_signal):
            self.input_signal = input_signal
        else:
            self.input_signal = lambda t: torch.tensor(float(input_signal), dtype=t.dtype, device=t.device)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the 2-variable system.

        Args:
            t: Time tensor
            x: State tensor [x, y]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [alpha, delta, beta, gamma]
            control: Control tensor [u_alpha, u_delta, u_beta, u_gamma]
                    multiplying each base parameter. Defaults to all 1.0.

        Returns:
            dx/dt tensor [dx/dt, dy/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 4, "Expected 4 fixed params"
        assert len(x) == 2, "Expected state [x, y]"

        # Unpack state
        x_state, y_state = x[0], x[1]

        # Get control (default to all 1.0)
        if control is not None:
            assert len(control) >= 4, "Expected 4 control inputs"
            u = control[:4]
        else:
            u = torch.ones(4, dtype=x.dtype, device=x.device)

        # Apply control to base parameters
        alpha = u[0] * fixed_params[0]
        delta = u[1] * fixed_params[1]
        beta = u[2] * fixed_params[2]
        gamma = u[3] * fixed_params[3]

        # Evaluate input signal at time t (fixed, not controllable)
        Input = self.input_signal(t)

        # Compute derivatives
        # dx/dt = alpha * Input(t) - delta * x
        dx_dt = alpha * Input - delta * x_state

        # dy/dt = beta * Input(t) - gamma * x * y
        dy_dt = beta * Input - gamma * x_state * y_state

        return torch.stack([dx_dt, dy_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"State Variables:\n"
            f"  x, y: State variables\n\n"
            f"Equations:\n"
            f"  dx/dt = (u[0] * alpha) * Input(t) - (u[1] * delta) * x\n"
            f"  dy/dt = (u[2] * beta) * Input(t) - (u[3] * gamma) * x * y\n\n"
            f"Base Parameters:\n"
            f"  alpha = {params[0]:.2f}  (input effect on x)\n"
            f"  delta = {params[1]:.2f}  (degradation rate of x)\n"
            f"  beta  = {params[2]:.2f}  (input effect on y)\n"
            f"  gamma = {params[3]:.2f}  (interaction rate x*y)\n\n"
            f"External Input Signal (not controllable):\n"
            f"  Input(t): Time-dependent signal (default: constant 1.0)\n\n"
            f"Control Inputs (4 total, all default to 1.0):\n"
            f"  u[0]: multiplier for alpha\n"
            f"  u[1]: multiplier for delta\n"
            f"  u[2]: multiplier for beta\n"
            f"  u[3]: multiplier for gamma\n"
            f"  When u=[1,1,1,1], behavior matches base system"
        )
