import torch
from typing import Callable
from ..base import ODE


class IFFL2Vars(ODE):
    """Simple 2-variable system with input control.

    A simple dynamical system where control modulates the input signal.

    State Variables:
        x: First state variable
        y: Second state variable

    Equations:
        dx/dt = alpha * (control * Input(t)) - delta * x
        dy/dt = beta * (control * Input(t)) - gamma * x * y

    Fixed Parameters (4 total):
        alpha: Input effect on x (default: 1.0)
        delta: Degradation rate of x (default: 0.1)
        beta: Input effect on y (default: 1.0)
        gamma: Interaction rate between x and y (default: 1.0)

    External Input Signal:
        Input(t): Time-dependent signal (default: constant 1.0)
                  Defined in the config file, evaluated at time t

    Control Input (scalar):
        control[0]: Multiplier for Input signal (default: 1.0)
                   Modulates the effect of Input on both equations
    """

    name = "2-Variable System with Input Control"
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
            # Default parameters
            fixed_params = torch.tensor([
                1.0,   # alpha: Input effect on x
                0.1,   # delta: Degradation rate of x
                1.0,   # beta: Input effect on y
                1.0,   # gamma: Interaction rate between x and y
            ])

        super().__init__(differentiable_params=None, fixed_params=fixed_params)

        # Store input signal (function or constant)
        if callable(input_signal):
            self.input_signal = input_signal
        else:
            # Convert constant to callable
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
            control: Control scalar that multiplies Input signal
                    Defaults to 1.0 if not provided.

        Returns:
            dx/dt tensor [dx/dt, dy/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 4, "Expected 4 fixed params"
        assert len(x) == 2, "Expected state [x, y]"

        # Unpack state
        x_state, y_state = x[0], x[1]

        # Unpack fixed parameters
        alpha, delta, beta, gamma = fixed_params

        # Get control (default to 1.0)
        if control is not None and len(control) > 0:
            u = control[0]
        else:
            u = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        # Evaluate input signal at time t
        Input = self.input_signal(t)

        # Apply control to input
        controlled_input = u * Input

        # Compute derivatives
        # dx/dt = alpha * (control * Input) - delta * x
        dx_dt = alpha * controlled_input - delta * x_state

        # dy/dt = beta * (control * Input) - gamma * x * y
        dy_dt = beta * controlled_input - gamma * x_state * y_state

        return torch.stack([dx_dt, dy_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"State Variables:\n"
            f"  x, y: State variables\n\n"
            f"Equations:\n"
            f"  dx/dt = alpha * (control * Input(t)) - delta * x\n"
            f"  dy/dt = beta * (control * Input(t)) - gamma * x * y\n\n"
            f"Parameters:\n"
            f"  alpha = {params[0]:.2f}  (input effect on x)\n"
            f"  delta = {params[1]:.2f}  (degradation rate of x)\n"
            f"  beta  = {params[2]:.2f}  (input effect on y)\n"
            f"  gamma = {params[3]:.2f}  (interaction rate x*y)\n\n"
            f"Input Signal:\n"
            f"  Input(t): Time-dependent signal (default: constant 1.0)\n\n"
            f"Control Input:\n"
            f"  control[0]: Scalar multiplier for Input (default: 1.0)\n"
            f"  Modulates the effect of Input on both equations"
        )
