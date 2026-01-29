# RPA Simulators

PyTorch-based library for differentiable ODE simulations, designed for reinforcement learning and optimal control of biological systems.

## Overview

This library provides differentiable ODE implementations of various control systems, with a focus on:
- **Gradient flow**: All systems use PyTorch for automatic differentiation
- **Control-modulated parameters**: Parameters can be dynamically adjusted via control inputs
- **External disturbances**: Separate time-varying external inputs (not controllable)
- **Flexible architecture**: Easy to add new systems following established patterns

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from rpasim.ode import HPA
from rpasim.plot.ode import plot_trajectory
import matplotlib.pyplot as plt

# Create HPA stress response system
hpa = HPA(stressor=lambda t: 1.0)

# Initial state: [CRH, ACTH, Cortisol, Pituitary, Adrenal]
x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# Plot 50-day trajectory
fig, axes = plot_trajectory(hpa, x0, T=50.0, n_steps=1000)
plt.show()
```

## Available Systems

### RPA Biological Control Systems (`rpasim.ode.rpa`)

#### AB Systems
- **`AB`**: Minimal integral feedback model (2 states)
  - Original version with `differentiable_params` and `fixed_params`
- **`ABControlled`**: Control-modulated version (2 states, 3 controls)
  - All parameters can be modulated: `dA/dt = α₁*u[0] + α₂*u[1]*A + α₃*u[2]*B`

#### HPA Systems (Stress Response)
- **`HPA`**: Full hypothalamic-pituitary-adrenal axis (5 states, 9 controls)
  - States: CRH, ACTH, Cortisol, Pituitary mass, Adrenal mass
  - External input: `I(t)` (stressor, not controllable)
  - Controls: I1-I3 (synthesis inhibitors), C1-C3 (receptor antagonists), A1-A3 (neutralizing antibodies)
  - Time unit: days

- **`HPASimple`**: Simplified version without gland dynamics (3 states, 9 controls)
  - Assumes constant gland masses (P=1, A=1)
  - Faster simulation for hormone-only dynamics

#### Network Motifs
- **`NFL`**: Negative Feedback Loop (3 states, 14 controls)
  - Topology: I→A→C, C→B→C (feedback)
  - Saturation conditions: `(1-B) >> K_CB`, `B >> K'_F_BB`
  - All parameters control-modulated

- **`IFFL`**: Incoherent Feed-Forward Loop (3 states, 14 controls)
  - Topology: I→A→B→C, A→C (competing pathways)
  - Saturation conditions: `(1-B) >> K_AB`, `B << K'_F_BB`
  - Demonstrates adaptation and pulse detection

### Classic Control ODEs (`rpasim.ode.classic_control`)

From Brunton et al., "Sparse identification of nonlinear dynamics with control (SINDy-MPC)", 2018:

- **`Lorenz`**: Chaotic convection system (3 states, 1 control)
- **`PopulationDynamics`**: Lotka-Volterra predator-prey (2 states, 1 control)
- **`FlightControl`**: F-8 Crusader aircraft (3 states, 1 control)
- **`HIVTreatment`**: HIV infection dynamics (5 states, 1 control)

## Key Concepts

### Parameter Types

| Type | Purpose | Gradients | Example |
|------|---------|-----------|---------|
| `fixed_params` | Physical constants | No | Degradation rates, affinity constants |
| `differentiable_params` | Learnable parameters | Yes | (Deprecated: use control-modulated instead) |
| `control` | Time-varying input | Runtime | Drug doses, control multipliers |
| External input | Non-controllable disturbance | Runtime | Stress level `u(t)`, environmental input |

### Control-Modulated Parameters

Modern systems (ABControlled, NFL, IFFL) use **control-modulated parameters**:
- Base parameters are fixed
- Control inputs multiply base parameters
- Default control = all 1.0 recovers base behavior
- Example: `effective_param = base_param * control[i]`

### External Inputs

Systems like HPA and NFL separate **external disturbances** from **control**:
- External input (e.g., stressor `I(t)`): Not controllable, represents environment
- Control (e.g., drug doses): Controllable, represents intervention
- Set via callable: `HPA(stressor=lambda t: 2.0 if t > 50 else 1.0)`

## Usage Examples

### Simulating with Control

```python
import torch
from torchdiffeq import odeint
from rpasim.ode import NFL

# Create NFL system
nfl = NFL(external_input=lambda t: 1.0)

# Define time-varying control (drug intervention)
def get_control(t):
    if t < 30:
        return torch.ones(14)  # No intervention
    else:
        # 50% inhibition of F_B at t=30
        u = torch.ones(14)
        u[13] = 0.5
        return u

# Wrapper for control
class ControlledODE:
    def __init__(self, ode, control_func):
        self.ode = ode
        self.control_func = control_func

    def __call__(self, t, x):
        return self.ode(t, x, control=self.control_func(t.item()))

nfl_ctrl = ControlledODE(nfl, get_control)

# Simulate
x0 = torch.tensor([0.5, 0.5, 0.5])
t = torch.linspace(0, 50.0, 1000)
trajectory = odeint(nfl_ctrl, x0, t, method='dopri5')
```

### Using the Environment Wrapper

```python
from rpasim.env.base import DifferentiableEnv
from rpasim.ode import Lorenz

# Create environment
env = DifferentiableEnv(
    reward_fn=lambda t, x: -torch.sum(x**2),  # Minimize state
    time_grid=torch.linspace(0, 10, 100),
)

# Reset with ODE and initial state
lorenz = Lorenz()
x0 = torch.tensor([1.0, 1.0, 1.0])
obs = env.reset(lorenz, x0)

# Take action (change to new ODE with control)
control = torch.tensor([2.0])
new_ode = Lorenz()  # In practice, you'd create with control
obs, reward, done, info = env.step((new_ode, 1.0))
```

## Project Structure

```
rpa-simulators/
├── src/rpasim/
│   ├── ode/
│   │   ├── base.py              # Abstract ODE base class
│   │   ├── rpa/                 # RPA biological systems
│   │   │   ├── ab.py           # AB systems
│   │   │   ├── hpa.py          # HPA stress response
│   │   │   ├── nfl.py          # Negative feedback loop
│   │   │   └── iffl.py         # Incoherent feed-forward
│   │   └── classic_control/    # Classic control benchmarks
│   │       ├── lorenz.py
│   │       ├── population.py
│   │       ├── flight.py
│   │       └── hiv.py
│   ├── env/
│   │   └── base.py             # DifferentiableEnv
│   ├── plot/
│   │   ├── ode.py              # ODE trajectory plotting
│   │   └── env.py              # Environment trajectory plotting
│   └── style.py                # Matplotlib styling
├── tests/                       # pytest test suite
├── notebooks/                   # Jupyter examples
└── scripts/                     # Utility scripts
```

## Design Patterns

### Creating a New ODE System

```python
import torch
from rpasim.ode.base import ODE

class MySystem(ODE):
    name = "My System"
    variable_names = ["x1", "x2"]
    fixed_param_names = ["k1", "k2"]

    def __init__(self, fixed_params=None):
        if fixed_params is None:
            fixed_params = torch.tensor([1.0, 0.5])
        super().__init__(differentiable_params=None, fixed_params=fixed_params)

    def forward(self, t, x, differentiable_params=None,
                fixed_params=None, control=None):
        # Unpack state and parameters
        x1, x2 = x[0], x[1]
        k1, k2 = fixed_params[0], fixed_params[1]

        # Extract control (default to 1.0)
        u = control[0] if control is not None else torch.tensor(1.0)

        # Compute derivatives
        dx1_dt = k1 * u * x1
        dx2_dt = -k2 * x2

        return torch.stack([dx1_dt, dx2_dt])

    def __str__(self):
        return f"{self.name}\nParameters: {self.fixed_params}"
```

## Contributing

When adding new systems:
1. Follow the established pattern (see `Design Patterns` above)
2. Include docstrings with equations and parameter descriptions
3. Add tests in `tests/`
4. Update this README

## References

- Brunton et al. (2018). "Sparse identification of nonlinear dynamics with control (SINDy-MPC)". *Proc. R. Soc. A* 474: 20180335.
- Karin et al. (2020). HPA axis modeling papers.
- Andersen et al. (2013). Hormone kinetics parameters.

## License

[Add license information]
