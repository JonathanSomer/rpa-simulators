import copy
from typing import Callable, Optional, Tuple, Union, List

import gymnasium
from gymnasium import spaces
import numpy as np
import torch

from rpasim.env import DifferentiableEnv
from rpasim.ode.rpa.ab import ABControlled


class ABGym(gymnasium.Env):
    """Gymnasium environment wrapping the ABControlled ODE.

    Action space: scalar u in [action_low, action_high].
    Each step copies the base ODE, sets alpha <- u * base_alpha,
    and advances the DifferentiableEnv by dt.

    Observation space: [A, B] state vector (float32).
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
        ode_method: str = "dopri5",
    ):
        super().__init__()

        assert action_low < action_high, "action_low must be less than action_high"
        reward_dt = time_horizon / n_reward_steps
        assert reward_dt < dt, (
            f"Reward grid spacing ({reward_dt:.4f}) must be smaller than dt ({dt}). "
            f"Increase n_reward_steps (currently {n_reward_steps}) so that "
            f"multiple reward samples fall within each dt step."
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(action_low), high=np.float32(action_high),
            shape=(1,), dtype=np.float32,
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
            ode_method=ode_method,
        )

    def _make_ode(self, u: float) -> ABControlled:
        """Create a new ABControlled with alpha <- u * base_alpha."""
        new_ode = copy.copy(self.base_ode)
        new_ode.fixed_params = torch.tensor([u * self.base_alpha])
        return new_ode

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment.

        Returns:
            obs: numpy array [A, B] (float32)
            info: dict
        """
        super().reset(seed=seed, options=options)
        (ode, state), info = self.env.reset(seed=seed, options=options)
        return state.detach().cpu().numpy().astype(np.float32), info

    def step(self, action: np.ndarray):
        """Take one step with action.

        Args:
            action: numpy array of shape (1,) with u in [action_low, action_high]

        Returns:
            obs: numpy array [A, B] (float32)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        u = float(action[0])
        u = np.clip(u, self.action_low, self.action_high)

        new_ode = self._make_ode(u)
        env_action = (new_ode, self.dt)
        (ode, state), reward, terminated, truncated, info = self.env.step(env_action)

        obs = state.detach().cpu().numpy().astype(np.float32)
        reward = float(reward)
        return obs, reward, terminated, truncated, info

    def get_trajectory(self):
        """Get the full trajectory so far.

        Returns:
            times, states, rewards tensors (torch)
        """
        return self.env.get_trajectory()

    @property
    def current_time(self) -> float:
        return self.env.current_time

    def close(self):
        self.env.close()
