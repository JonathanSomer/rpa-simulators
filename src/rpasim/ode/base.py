from abc import ABC, abstractmethod
from typing import Optional, List
import torch


class ODE(ABC):
    """Base class for ODE systems.

    All inputs and outputs are PyTorch tensors.
    Subclasses must implement the forward() method to compute dx/dt.

    Attributes:
        variable_names: List of variable names (e.g., ["A", "B"])
    """

    variable_names: List[str] = []

    def __init__(
        self,
        differentiable_params: Optional[torch.Tensor] = None,
        fixed_params: Optional[torch.Tensor] = None,
    ):
        """Initialize ODE with parameters.

        Args:
            differentiable_params: Parameters that require gradients
            fixed_params: Fixed parameters (no gradients)
        """
        self.differentiable_params = differentiable_params
        self.fixed_params = fixed_params

    def set_parameters(self, differentiable_params: torch.Tensor):
        """Update differentiable parameters.

        Args:
            differentiable_params: New differentiable parameters
        """
        self.differentiable_params = differentiable_params

    @abstractmethod
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: Optional[torch.Tensor] = None,
        fixed_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dx/dt.

        Args:
            t: Time tensor
            x: State tensor
            differentiable_params: Optional differentiable parameters
            fixed_params: Optional fixed parameters

        Returns:
            dx/dt tensor
        """
        pass

    def __call__(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: Optional[torch.Tensor] = None,
        fixed_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dx/dt (makes the class callable).

        Uses params from __init__ as defaults if not provided.

        Args:
            t: Time tensor
            x: State tensor
            differentiable_params: Optional override for differentiable params
            fixed_params: Optional override for fixed params

        Returns:
            dx/dt tensor
        """
        # Use provided params or fall back to instance params
        diff_params = (
            differentiable_params
            if differentiable_params is not None
            else self.differentiable_params
        )
        fix_params = (
            fixed_params
            if fixed_params is not None
            else self.fixed_params
        )

        return self.forward(t, x, diff_params, fix_params)
