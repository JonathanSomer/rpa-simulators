import torch
from ..base import ODE


class HIVTreatment(ODE):
    """HIV/AIDS treatment ODE with immune response and HAART therapy.

    From Brunton et al., "Sparse identification of nonlinear dynamics with
    control (SINDy-MPC)", Proc. R. Soc. A 474: 20180335 (2018).

    Models the interaction between HIV and CD4+ cells with highly active
    anti-retroviral therapy (HAART). The system tracks healthy cells,
    infected cells, and various cytotoxic lymphocyte (CTL) populations.

    Equations (uncontrolled, u=0):
        dx1/dt = λ - d*x1 - β*x1*x2
        dx2/dt = β*x1*x2 - a*x2 - p1*x4*x2 - p2*x5*x2
        dx3/dt = c2*x1*x2*x3 - c2*q*x2*x3 - b2*x3
        dx4/dt = c1*x2*x4 - b1*x4
        dx5/dt = c2*q*x2*x3 - h*x5

    Control effects (when u != 0):
        The control u affects the infection rate: β_eff = β(1 - η*u)
        This modifies x1 and x2 equations:
        dx1/dt = λ - d*x1 - β(1 - η*u)*x1*x2
        dx2/dt = β(1 - η*u)*x1*x2 - a*x2 - p1*x4*x2 - p2*x5*x2

    Parameters (all fixed):
        λ = 1.0     (production rate of healthy cells)
        d = 0.1     (death rate of healthy cells)
        β = 1.0     (infection rate)
        a = 0.2     (death rate of infected cells)
        p1 = 1.0    (killing rate by helper-independent CTL)
        p2 = 1.0    (killing rate by helper-dependent CTL)
        c1 = 0.03   (proliferation rate of helper-independent CTL)
        c2 = 0.06   (proliferation rate of CTL precursors)
        b1 = 0.1    (death rate of helper-independent CTL)
        b2 = 0.01   (death rate of CTL precursors)
        q = 0.5     (conversion rate to helper-dependent CTL)
        h = 0.1     (death rate of helper-dependent CTL)
        η = 0.9799  (efficacy of HAART therapy)

    Control objective:
        - Cost: J = ∫(x1 - x̂1) + (x3 - x̂3) + |u| dt
        - where x̂1, x̂3 are healthy steady state values
        - Control limits: u ∈ [0, 1]
        - Time horizon: 50 weeks
        - MPC horizon: 2 days (24 time steps of 2h each)

    State:
        x1: healthy CD4+ T-cells
        x2: HIV-infected CD4+ T-cells
        x3: CTL precursors (memory CTL)
        x4: helper-independent CTL
        x5: helper-dependent CTL
    """

    name = "HIV/AIDS Treatment"
    variable_names = ["healthy_CD4", "infected_CD4", "CTL_precursor", "CTL_indep", "CTL_dep"]
    fixed_param_names = ["lambda", "d", "beta", "a", "p1", "p2", "c1", "c2", "b1", "b2", "q", "h", "eta"]

    def __init__(self):
        """Initialize HIV treatment ODE with fixed parameters.

        Fixed parameters: [λ, d, β, a, p1, p2, c1, c2, b1, b2, q, h, η]
        """
        # Fixed parameters
        fixed_params = torch.tensor([
            1.0,      # λ (lambda)
            0.1,      # d
            1.0,      # β (beta)
            0.2,      # a
            1.0,      # p1
            1.0,      # p2
            0.03,     # c1
            0.06,     # c2
            0.1,      # b1
            0.01,     # b2
            0.5,      # q
            0.1,      # h
            0.9799    # η (eta)
        ])
        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the HIV treatment system.

        Args:
            t: Time tensor (unused in this system)
            x: State tensor [x1, x2, x3, x4, x5]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [λ, d, β, a, p1, p2, c1, c2, b1, b2, q, h, η]
            control: Control input u (HAART therapy level) in [0, 1]
                    Can be scalar or tensor of shape (1,) for single control input

        Returns:
            dx/dt tensor [dx1/dt, dx2/dt, dx3/dt, dx4/dt, dx5/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 13, "Expected 13 fixed params"
        assert len(x) == 5, "Expected state [x1, x2, x3, x4, x5]"

        # Unpack state
        x1, x2, x3, x4, x5 = x

        # Unpack parameters
        lambda_p, d, beta, a, p1, p2, c1, c2, b1, b2, q, h, eta = fixed_params

        # Extract control input (default to 0 if not provided)
        if control is not None:
            u = control[0] if control.dim() > 0 else control
        else:
            u = torch.tensor(0.0)

        # Effective infection rate with control: β_eff = β(1 - η*u)
        beta_eff = beta * (1 - eta * u)

        # Compute derivatives with control
        dx1_dt = lambda_p - d * x1 - beta_eff * x1 * x2
        dx2_dt = beta_eff * x1 * x2 - a * x2 - p1 * x4 * x2 - p2 * x5 * x2
        dx3_dt = c2 * x1 * x2 * x3 - c2 * q * x2 * x3 - b2 * x3
        dx4_dt = c1 * x2 * x4 - b1 * x4
        dx5_dt = c2 * q * x2 * x3 - h * x5

        return torch.stack([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params
        lambda_p, d, beta, a, p1, p2, c1, c2, b1, b2, q, h, eta = params

        return (
            f"{self.name}\n\n"
            f"Equations (uncontrolled, u=0):\n"
            f"  dx1/dt = {lambda_p:.2f} - {d:.2f}*x1 - {beta:.2f}*x1*x2\n"
            f"  dx2/dt = {beta:.2f}*x1*x2 - {a:.2f}*x2 - {p1:.2f}*x4*x2 - {p2:.2f}*x5*x2\n"
            f"  dx3/dt = {c2:.2f}*x1*x2*x3 - {c2:.2f}*{q:.2f}*x2*x3 - {b2:.2f}*x3\n"
            f"  dx4/dt = {c1:.2f}*x2*x4 - {b1:.2f}*x4\n"
            f"  dx5/dt = {c2:.2f}*{q:.2f}*x2*x3 - {h:.2f}*x5\n"
            f"Control: u ∈ [0, 1], efficacy η = {eta:.4f}, horizon = 50 weeks"
        )
