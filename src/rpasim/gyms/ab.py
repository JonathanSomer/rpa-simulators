import copy
from typing import Callable, Optional, Tuple, Union, List

import torch

from rpasim.env import DifferentiableEnv
from rpasim.ode.rpa.ab import ABControlled


class ABGym:
    """Gymnasium-style wrapper for the ABControlled ODE.

    Action space: scalar u in [action_low, action_high].
    Each step copies the base ODE, sets alpha <- u * base_alpha,
    and advances the DifferentiableEnv by dt.
    """

    def __init__(
        self,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        time_horizon: float = 10.0,
        dt: float = 0.1,
        n_reward_steps: int = 300,
        action_low: float = 0.1,
        action_high: float = 1.0,
        alpha: float = 50.0,
        state_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        initial_state_range: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    ):
        assert action_low < action_high, "action_low must be less than action_high"
        reward_dt = time_horizon / n_reward_steps
        assert reward_dt < dt, (
            f"Reward grid spacing ({reward_dt:.4f}) must be smaller than dt ({dt}). "
            f"Increase n_reward_steps (currently {n_reward_steps}) so that "
            f"multiple reward samples fall within each dt step."
        )

        self.base_ode = ABControlled(fixed_params=torch.tensor([alpha]))
        self.current_ode = self.base_ode  # for compatibility with plotting functions
        self.base_alpha = alpha
        self.dt = dt
        self.action_low = action_low
        self.action_high = action_high

        self.env = DifferentiableEnv(
            initial_ode=self.base_ode,
            reward_fn=reward_fn,
            initial_state=initial_state,
            time_horizon=time_horizon,
            n_reward_steps=n_reward_steps,
            state_limits=state_limits,
            initial_state_range=initial_state_range,
        )

    def _make_ode(self, u: float) -> ABControlled:
        """Create a new ABControlled with alpha <- u * base_alpha."""
        new_ode = copy.copy(self.base_ode)
        new_ode.fixed_params = torch.tensor([u * self.base_alpha])
        return new_ode

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment.

        Returns:
            state: current state tensor [A, B]
            info: dict
        """
        (ode, state), info = self.env.reset(seed=seed, options=options)
        return state, info

    def step(self, u: float):
        """Take one step with action u.

        Args:
            u: control input in [action_low, action_high]

        Returns:
            state: new state tensor [A, B]
            reward: scalar reward
            terminated: bool
            truncated: bool
            info: dict
        """
        assert (
            self.action_low <= u <= self.action_high
        ), f"Action u={u} out of range [{self.action_low}, {self.action_high}]"

        new_ode = self._make_ode(u)
        action = (new_ode, self.dt)
        (ode, state), reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info

    def get_trajectory(self):
        """Get the full trajectory so far.

        Returns:
            times, states, rewards tensors
        """
        return self.env.get_trajectory()

    @property
    def current_time(self) -> float:
        return self.env.current_time

    def close(self):
        self.env.close()
