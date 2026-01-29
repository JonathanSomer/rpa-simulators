import torch
from typing import Callable
from ..base import ODE


class IFFL(ODE):
    """Incoherent Feed-Forward Loop (IFFL) system with control-modulated parameters.

    A 3-node feed-forward network with both direct and indirect pathways.
    All parameters can be modulated by control inputs.

    State Variables:
        A: Node A concentration
        B: Node B concentration
        C: Node C concentration

    Equations:
        dA/dt = I * k_IA * (1-A)/((1-A)+K_IA) - F_A * k'_F_AA * A/(A+K'_F_AA)
        dB/dt = A * k_AB * (1-B)/((1-B)+K_AB) - F_B * k'_F_BB * B/(B+K'_F_BB)
        dC/dt = A * k_AC * (1-C)/((1-C)+K_AC) - B * k'_BC * C/(C+K'_BC)

    Where each parameter is multiplied by its corresponding control input u[i].

    Fixed Parameters (14 total):
        k_IA, K_IA: Activation of A by I (rate and affinity)
        k'_F_AA, K'_F_AA: Degradation of A by F_A (rate and affinity)
        k_AB, K_AB: Activation of B by A (rate and affinity)
        k'_F_BB, K'_F_BB: Degradation of B by F_B (rate and affinity)
        k_AC, K_AC: Activation of C by A (rate and affinity)
        k'_BC, K'_BC: Degradation of C by B (rate and affinity)
        F_A: Basal enzyme concentration for A
        F_B: Basal enzyme concentration for B

    Saturation conditions for B* proportional to A*:
        (1-B) >> K_AB  (K_AB very small)
        B << K'_F_BB   (K'_F_BB very large)

    External Input (not controllable):
        I(t): External input function (default: constant 1.0)

    Control Inputs (14 total, all default to 1.0):
        u[0..13]: Multipliers for each of the 14 fixed parameters
        When all u[i] = 1.0, behavior matches the base system

    Control vector order: [u_k_IA, u_K_IA, u_k'_F_AA, u_K'_F_AA, u_k_AB, u_K_AB,
                          u_k'_F_BB, u_K'_F_BB, u_k_AC, u_K_AC, u_k'_BC, u_K'_BC,
                          u_F_A, u_F_B]
    """

    name = "Incoherent Feed-Forward Loop (Controlled)"
    variable_names = ["A", "B", "C"]
    fixed_param_names = [
        "k_IA", "K_IA",
        "k'_F_AA", "K'_F_AA",
        "k_AB", "K_AB",
        "k'_F_BB", "K'_F_BB",
        "k_AC", "K_AC",
        "k'_BC", "K'_BC",
        "F_A", "F_B",
    ]

    def __init__(
        self,
        fixed_params: torch.Tensor | None = None,
        external_input: Callable[[float], float] | None = None,
    ):
        """Initialize IFFL ODE with parameters.

        Args:
            fixed_params: [k_IA, K_IA, k'_F_AA, K'_F_AA, k_AB, K_AB,
                          k'_F_BB, K'_F_BB, k_AC, K_AC, k'_BC, K'_BC,
                          F_A, F_B]
                         Uses defaults if not provided.
            external_input: Function I(t) representing external input.
                           Defaults to constant 1.0.
        """
        if fixed_params is None:
            # Default parameters satisfying (1-B) >> K_AB and B << K'_F_BB
            fixed_params = torch.tensor([
                1.0,    # k_IA: activation rate of A by I
                0.5,    # K_IA: affinity constant for A activation
                1.0,    # k'_F_AA: degradation rate of A by F_A
                0.5,    # K'_F_AA: affinity constant for A degradation
                1.0,    # k_AB: activation rate of B by A
                0.001, # K_AB: very small (so (1-B) >> K_AB)
                1.0,    # k'_F_BB: degradation rate of B by F_B
                100.0,   # K'_F_BB: very large (so B << K'_F_BB)
                1.0,    # k_AC: activation rate of C by A
                0.5,    # K_AC: affinity constant for C activation
                1.0,    # k'_BC: degradation rate of C by B
                0.5,    # K'_BC: affinity constant for C degradation
                1.0,    # F_A: basal enzyme for A
                200.0,    # F_B: basal enzyme for B
            ])

        super().__init__(differentiable_params=None, fixed_params=fixed_params)

        # External input function (not controllable)
        self.external_input = external_input if external_input is not None else lambda t: 1.0

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        differentiable_params: torch.Tensor | None = None,
        fixed_params: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dx/dt for the IFFL system.

        Args:
            t: Time tensor
            x: State tensor [A, B, C]
            differentiable_params: Not used (all params are fixed)
            fixed_params: [k_IA, K_IA, k'_F_AA, K'_F_AA, k_AB, K_AB,
                          k'_F_BB, K'_F_BB, k_AC, K_AC, k'_BC, K'_BC,
                          F_A, F_B]
            control: Control tensor [u_0, u_1, ..., u_13] multiplying each parameter
                    Defaults to all 1.0 if not provided.

        Returns:
            dx/dt tensor [dA/dt, dB/dt, dC/dt]
        """
        assert fixed_params is not None, "Fixed params required"
        assert len(fixed_params) == 14, "Expected 14 fixed params"
        assert len(x) == 3, "Expected state [A, B, C]"

        # Unpack state
        A, B, C = x[0], x[1], x[2]

        # Unpack fixed parameters
        params = fixed_params

        # Extract control inputs (default to 1.0 if not provided)
        if control is not None:
            assert len(control) >= 14, "Expected 14 control inputs"
            u = control[:14]
        else:
            u = torch.ones(14, dtype=x.dtype)

        # Apply control to parameters
        k_IA = u[0] * params[0]
        K_IA = u[1] * params[1]
        k_prime_F_AA = u[2] * params[2]
        K_prime_F_AA = u[3] * params[3]
        k_AB = u[4] * params[4]
        K_AB = u[5] * params[5]
        k_prime_F_BB = u[6] * params[6]
        K_prime_F_BB = u[7] * params[7]
        k_AC = u[8] * params[8]
        K_AC = u[9] * params[9]
        k_prime_BC = u[10] * params[10]
        K_prime_BC = u[11] * params[11]
        F_A = u[12] * params[12]
        F_B = u[13] * params[13]

        # Get external input at current time
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        I = self.external_input(t_val)
        if not isinstance(I, torch.Tensor):
            I = torch.tensor(I, dtype=x.dtype)

        # Compute derivatives with control-modulated parameters
        # dA/dt = I * k_IA * (1-A)/((1-A)+K_IA) - F_A * k'_F_AA * A/(A+K'_F_AA)
        activation_A = I * k_IA * (1.0 - A) / ((1.0 - A) + K_IA)
        degradation_A = F_A * k_prime_F_AA * A / (A + K_prime_F_AA)
        dA_dt = activation_A - degradation_A

        # dB/dt = A * k_AB * (1-B)/((1-B)+K_AB) - F_B * k'_F_BB * B/(B+K'_F_BB)
        activation_B = A * k_AB * (1.0 - B) / ((1.0 - B) + K_AB)
        degradation_B = F_B * k_prime_F_BB * B / (B + K_prime_F_BB)
        dB_dt = activation_B - degradation_B

        # dC/dt = A * k_AC * (1-C)/((1-C)+K_AC) - B * k'_BC * C/(C+K'_BC)
        activation_C = A * k_AC * (1.0 - C) / ((1.0 - C) + K_AC)
        degradation_C = B * k_prime_BC * C / (C + K_prime_BC)
        dC_dt = activation_C - degradation_C

        return torch.stack([dA_dt, dB_dt, dC_dt])

    def __str__(self) -> str:
        """Return string representation with equations and parameters."""
        params = self.fixed_params

        return (
            f"{self.name}\n\n"
            f"State Variables:\n"
            f"  A, B, C: Node concentrations\n\n"
            f"Equations:\n"
            f"  dA/dt = I*k_IA*u[0]*(1-A)/((1-A)+K_IA*u[1]) - F_A*u[12]*k'_F_AA*u[2]*A/(A+K'_F_AA*u[3])\n"
            f"  dB/dt = A*k_AB*u[4]*(1-B)/((1-B)+K_AB*u[5]) - F_B*u[13]*k'_F_BB*u[6]*B/(B+K'_F_BB*u[7])\n"
            f"  dC/dt = A*k_AC*u[8]*(1-C)/((1-C)+K_AC*u[9]) - B*k'_BC*u[10]*C/(C+K'_BC*u[11])\n\n"
            f"Base Parameters:\n"
            f"  k_IA={params[0]:.2f}, K_IA={params[1]:.2f}\n"
            f"  k'_F_AA={params[2]:.2f}, K'_F_AA={params[3]:.2f}\n"
            f"  k_AB={params[4]:.2f}, K_AB={params[5]:.6f}\n"
            f"  k'_F_BB={params[6]:.2f}, K'_F_BB={params[7]:.2f}\n"
            f"  k_AC={params[8]:.2f}, K_AC={params[9]:.2f}\n"
            f"  k'_BC={params[10]:.2f}, K'_BC={params[11]:.2f}\n"
            f"  F_A={params[12]:.2f}, F_B={params[13]:.2f}\n\n"
            f"Saturation Conditions:\n"
            f"  (1-B) >> K_AB (K_AB very small)\n"
            f"  B << K'_F_BB (K'_F_BB large)\n\n"
            f"External Input:\n"
            f"  I(t): external input (not controllable)\n\n"
            f"Control (14 inputs, all default to 1.0):\n"
            f"  u[0..13]: multipliers for each parameter\n"
            f"  When u=[1,1,...,1], behavior matches base system"
        )
