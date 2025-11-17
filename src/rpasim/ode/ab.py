import torch
from .base import ODE


class AB(ODE):
    """Two-variable AB ODE system.

    Equations:
        dA/dt = alpha1 + alpha2*A + alpha3*B
        dB/dt = beta1*A - beta2**B

    Parameters:
        differentiable_params: [alpha1, alpha2, alpha3]
        fixed_params: [beta1, beta2]

    State:
        x: [A, B]
    """

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
            differentiable_params: [alpha1, alpha2, alpha3]
            fixed_params: [beta1, beta2]

        Returns:
            dx/dt tensor [dA/dt, dB/dt]
        """
        assert len(differentiable_params) == 3, "Expected 3 differentiable params"
        assert len(fixed_params) == 2, "Expected 2 fixed params"
        assert len(x) == 2, "Expected state [A, B]"

        # Unpack state
        A = x[0]
        B = x[1]

        # Unpack parameters
        alpha1, alpha2, alpha3 = differentiable_params
        beta1, beta2 = fixed_params

        # Compute derivatives
        dA_dt = alpha1 + alpha2 * A + alpha3 * B
        dB_dt = beta1 * A - beta2 * B

        return torch.stack([dA_dt, dB_dt])
