import torch
import numpy as np
from typing import Callable
from ..base import ODE


class HPA(ODE):
    """Hypothalamic-Pituitary-Adrenal (HPA) axis model with gland dynamics.

    Models the stress response system including hormone dynamics and gland mass.
    Based on Karin et al. and Andersen et al. models.

    State Variables:
        x1: CRH (Corticotropin-releasing hormone)
        x2: ACTH (Adrenocorticotropic hormone)
        x3: Cortisol hormone
        P: Pituitary corticotroph functional mass
        A: Adrenal functional mass

    Equations:
        dx1/dt = γ_x1 * (I1 * u(t) * MR(C3*x3) * GR(C3*x3) - A1 * x1)
        dx2/dt = γ_x2 * (I2 * (C1*x1) * P * GR(C3*x3) - A2 * x2)
        dx3/dt = γ_x3 * (I3 * (C2*x2) * A - A3 * x3)
        dP/dt = γ_P * P * ((C1*x1) * (1 - P/K_P) - 1)
        dA/dt = γ_A * A * ((C2*x2) * (1 - A/K_A) - 1)

    Where:
        MR(x) = 1/x  (Mineralocorticoid receptor function)
        GR(x) = 1 / ((x/K_GR)^n_GR + 1)  (Glucocorticoid receptor function)
        u(t) = stressor input (external, not controllable)

    Fixed Parameters:
        gamma_x1: CRH degradation rate [1/day]
        gamma_x2: ACTH degradation rate [1/day]
        gamma_x3: Cortisol degradation rate [1/day]
        gamma_P: Pituitary turnover rate [1/day]
        gamma_A: Adrenal turnover rate [1/day]
        KGR: Glucocorticoid receptor dissociation constant
        nGR: Hill coefficient for GR
        KP: Pituitary carrying capacity
        KA: Adrenal carrying capacity

    External Input (not controllable):
        u(t): Stressor input function of time (default: constant 1.0)

    Control Inputs (9 total, all default to 1.0):
        I1, I2, I3: Synthesis inhibitors for CRH, ACTH, Cortisol [0, 1]
        C1, C2, C3: Receptor antagonists for CRH, ACTH, Cortisol [0, 1]
        A1, A2, A3: Neutralizing antibodies for CRH, ACTH, Cortisol [1, ∞)

    Control vector order: [I1, I2, I3, C1, C2, C3, A1, A2, A3]
    """

    name = "HPA Axis (Stress Response)"
    variable_names = ["CRH", "ACTH", "Cortisol", "Pituitary", "Adrenal"]
    fixed_param_names = [
        "gamma_x1", "gamma_x2", "gamma_x3",
        "gamma_P", "gamma_A",
        "KGR", "nGR", "KP", "KA"
    ]

    def __init__(
        self,
        fixed_params: torch.Tensor | None = None,
        stressor: Callable[[float], float] | None = None,
    ):
        """Initialize HPA ODE with parameters.

        Args:
            fixed_params: [gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A,
                          KGR, nGR, KP, KA]
                         Uses physiological defaults if not provided.
            stressor: Function u(t) representing external stressor input.
                     Defaults to constant 1.0.
        """
        if fixed_params is None:
            # Default parameters (time unit: days)
            # Converted from 1/min to 1/day where needed
            fixed_params = torch.tensor([
                np.log(2) / 4 * 24 * 60,    # gamma_x1: ~249.5 1/day (from 0.17 1/min)
                np.log(2) / 20 * 24 * 60,   # gamma_x2: ~49.9 1/day (from 0.035 1/min)
                np.log(2) / 80 * 24 * 60,   # gamma_x3: ~12.5 1/day (from 0.0086 1/min)
                np.log(2) / 20,              # gamma_P: ~0.035 1/day
                np.log(2) / 30,              # gamma_A: ~0.023 1/day
                4.0,                         # KGR
                3.0,                         # nGR
                1e6,                         # KP (large = no carrying capacity)
                1e6,                         # KA (large = no carrying capacity)
            ], dtype=torch.float32)

        super().__init__(differentiable_params=None, fixed_params=fixed_params)

        # External stressor function (not controllable)
        self.stressor = stressor if stressor is not None else lambda t: 1.0

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the HPA system.

        Args:
            t: Time tensor
            x: State tensor [CRH, ACTH, Cortisol, Pituitary, Adrenal]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A,
                          KGR, nGR, KP, KA]
            control: Control tensor [I1, I2, I3, C1, C2, C3, A1, A2, A3]
                    Defaults to [1, 1, 1, 1, 1, 1, 1, 1, 1] if not provided.

        Returns:
            dx/dt tensor [dCRH/dt, dACTH/dt, dCortisol/dt, dP/dt, dA/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 9, "Expected 9 fixed params"
        assert len(x) == 5, "Expected state [CRH, ACTH, Cortisol, P, A]"

        # Unpack state
        x1, x2, x3, P, A = x[0], x[1], x[2], x[3], x[4]

        # Unpack fixed parameters
        gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A, KGR, nGR, KP, KA = fixed_params

        # Get stressor value at current time
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        u = self.stressor(t_val)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=x.dtype)

        # Extract control inputs (default to 1.0 if not provided)
        if control is not None:
            assert len(control) >= 9, "Expected 9 control inputs [I1, I2, I3, C1, C2, C3, A1, A2, A3]"
            I1, I2, I3 = control[0], control[1], control[2]
            C1, C2, C3 = control[3], control[4], control[5]
            A1, A2, A3 = control[6], control[7], control[8]
        else:
            I1 = I2 = I3 = torch.tensor(1.0)
            C1 = C2 = C3 = torch.tensor(1.0)
            A1 = A2 = A3 = torch.tensor(1.0)

        # Receptor functions
        def MR(x_val):
            """Mineralocorticoid receptor function: MR(x) = 1/x"""
            return 1.0 / x_val

        def GR(x_val):
            """Glucocorticoid receptor function: GR(x) = 1 / ((x/K_GR)^n_GR + 1)"""
            return 1.0 / (torch.pow(x_val / KGR, nGR) + 1.0)

        # Effective cortisol seen by receptors (modulated by C3 antagonist)
        x3_eff = C3 * x3

        # Compute derivatives (from paper equations)
        # dx1 = gamma_x1 * (I1 * u * MR(C3*x3) * GR(C3*x3) - A1 * x1)
        dx1_dt = gamma_x1 * (I1 * u * MR(x3_eff) * GR(x3_eff) - A1 * x1)

        # dx2 = gamma_x2 * (I2 * (C1*x1) * P * GR(C3*x3) - A2 * x2)
        dx2_dt = gamma_x2 * (I2 * (C1 * x1) * P * GR(x3_eff) - A2 * x2)

        # dx3 = gamma_x3 * (I3 * (C2*x2) * A - A3 * x3)
        dx3_dt = gamma_x3 * (I3 * (C2 * x2) * A - A3 * x3)

        # dP = gamma_P * P * ((C1*x1) * (1 - P/KP) - 1)
        dP_dt = gamma_P * P * ((C1 * x1) * (1.0 - P / KP) - 1.0)

        # dA = gamma_A * A * ((C2*x2) * (1 - A/KA) - 1)
        dA_dt = gamma_A * A * ((C2 * x2) * (1.0 - A / KA) - 1.0)

        return torch.stack([dx1_dt, dx2_dt, dx3_dt, dP_dt, dA_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A, KGR, nGR, KP, KA = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"State Variables:\n"
            f"  x1: CRH, x2: ACTH, x3: Cortisol, P: Pituitary, A: Adrenal\n\n"
            f"Equations:\n"
            f"  dx1/dt = γ_x1 * (I1*u(t)*MR(C3*x3)*GR(C3*x3) - A1*x1)\n"
            f"  dx2/dt = γ_x2 * (I2*(C1*x1)*P*GR(C3*x3) - A2*x2)\n"
            f"  dx3/dt = γ_x3 * (I3*(C2*x2)*A - A3*x3)\n"
            f"  dP/dt = γ_P * P * ((C1*x1)*(1 - P/K_P) - 1)\n"
            f"  dA/dt = γ_A * A * ((C2*x2)*(1 - A/K_A) - 1)\n\n"
            f"Parameters:\n"
            f"  γ_x1 = {gamma_x1:.2f}, γ_x2 = {gamma_x2:.2f}, γ_x3 = {gamma_x3:.2f}\n"
            f"  γ_P = {gamma_P:.4f}, γ_A = {gamma_A:.4f}\n"
            f"  K_GR = {KGR:.1f}, n_GR = {nGR:.1f}\n"
            f"  K_P = {KP:.0e}, K_A = {KA:.0e}\n\n"
            f"External Input:\n"
            f"  u(t): stressor function (not controllable)\n\n"
            f"Control (9 inputs, all default to 1.0):\n"
            f"  I1-I3: synthesis inhibitors [0,1]\n"
            f"  C1-C3: receptor antagonists [0,1]\n"
            f"  A1-A3: neutralizing antibodies [1,∞)"
        )
