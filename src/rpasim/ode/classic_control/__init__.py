"""Classic control ODEs from Brunton et al. SINDy-MPC paper."""

from .population import PopulationDynamics
from .lorenz import Lorenz

__all__ = ["PopulationDynamics", "Lorenz"]
