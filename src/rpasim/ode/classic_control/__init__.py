"""Classic control ODEs from Brunton et al. SINDy-MPC paper."""

from .population import PopulationDynamics
from .lorenz import Lorenz
from .flight import FlightControl
from .hiv import HIVTreatment

__all__ = ["PopulationDynamics", "Lorenz", "FlightControl", "HIVTreatment"]
