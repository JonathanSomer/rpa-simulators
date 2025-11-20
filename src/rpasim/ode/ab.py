import torch
from .base import ODE


class AB(ODE):
    """Minimal integral feedback model with two variables.

    This is a minimal model for integral feedback control, where A acts as
    an integral feedback controller of B.

    Equations:
        dA/dt = alpha1 + alpha2*A + alpha3*B
        dB/dt = beta1*A - beta2*B

    Steady state (fixed points):
        B_ss = alpha1 / (-alpha3) = 4  (when alpha1=1, alpha3=-1/4)
        A_ss = beta2*B_ss / beta1 = 1  (when beta1=4, beta2=1, B_ss=4)

    Parameters:
        differentiable_params: [alpha1, alpha2, alpha3]
        fixed_params: [beta1, beta2]

    Default parameters:
        alpha1 = 1, alpha2 = 0, alpha3 = -1/4
        beta1 = 4, beta2 = 1

    State:
        x: [A, B]
    """

    name = "AB (Integral Feedback)"
    variable_names = ["A", "B"]
    differentiable_param_names = ["alpha1", "alpha2", "alpha3"]
    fixed_param_names = ["beta1", "beta2"]

    def __init__(
        self,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
    ):
        """Initialize AB ODE with parameters.

        Args:
            differentiable_params: [alpha1, alpha2, alpha3]. Defaults to [1, 0, -1/4]
            fixed_params: [beta1, beta2]. Defaults to [4, 1]
        """
        if differentiable_params is None:
            differentiable_params = torch.tensor([1.0, 0.0, -1/4])
        if fixed_params is None:
            fixed_params = torch.tensor([4.0, 1.0])

        super().__init__(
            differentiable_params=differentiable_params,
            fixed_params=fixed_params
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor,
        fixed_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dx/dt for the AB ODE system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [A, B]
            differentiable_params: [alpha1, alpha2, alpha3] OR [alpha2, alpha3] if alpha1 is fixed
            fixed_params: [beta1, beta2] OR [alpha1, beta1, beta2] if alpha1 is fixed

        Returns:
            dx/dt tensor [dA/dt, dB/dt]
        """
        assert len(x) == 2, "Expected state [A, B]"

        # Unpack state
        A = x[0]
        B = x[1]

        # Flexible parameter handling
        if len(differentiable_params) == 3 and len(fixed_params) == 2:
            # Standard: [alpha1, alpha2, alpha3] differentiable, [beta1, beta2] fixed
            alpha1, alpha2, alpha3 = differentiable_params
            beta1, beta2 = fixed_params
        elif len(differentiable_params) == 2 and len(fixed_params) == 3:
            # Alternative: [alpha2, alpha3] differentiable, [alpha1, beta1, beta2] fixed
            alpha2, alpha3 = differentiable_params
            alpha1, beta1, beta2 = fixed_params
        else:
            raise ValueError(f"Unsupported parameter configuration: {len(differentiable_params)} differentiable, {len(fixed_params)} fixed")

        # Compute derivatives
        dA_dt = alpha1 + alpha2 * A + alpha3 * B
        dB_dt = beta1 * A - beta2 * B

        return torch.stack([dA_dt, dB_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        alpha1, alpha2, alpha3 = self.differentiable_params
        beta1, beta2 = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"Equations:\n"
            f"  dA/dt = {alpha1:.2f} + {alpha2:.2f}*A + {alpha3:.2f}*B\n"
            f"  dB/dt = {beta1:.2f}*A - {beta2:.2f}*B\n"
            f"Parameters:\n"
            f"  alpha1 = {alpha1:.2f}, alpha2 = {alpha2:.2f}, alpha3 = {alpha3:.2f}\n"
            f"  beta1 = {beta1:.2f}, beta2 = {beta2:.2f}"
        )
