import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class HerReplayBufferExt(HerReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    The main differences between this implementation of HER and the stable-baselines3 implementation are:
        - this version can also relabel desired goals in the info dictionary

    .. note::

      Compared to other implementations, the ``future`` goal sampling strategy is inclusive:
      the current transition can be used when re-sampling.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param copy_info_dict: Whether to copy the info dictionary and pass it to
        ``compute_reward()`` method. False by default.
        Please note that the copy may cause a slowdown. If "achieved_goal" and "desired_goal"
        are keys in the info dictionary not only ``obs["desired_goal"]``, but also ``info["desired_goal"]``
        will be relabeled.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        env: VecEnv,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        copy_info_dict: bool = False,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            env=env,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            copy_info_dict=copy_info_dict,
        )

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals, infos = self._sample_goals(batch_indices, env_indices, infos)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )

    def _sample_goals(
        self, batch_indices: np.ndarray, env_indices: np.ndarray, batch_infos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample goals based on goal_selection_strategy.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param batch_infos: info dictionaries of the transitions
        :return: Sampled goals, relabeled info dictionaries
        """
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Replay with random state which comes from the same episode and was observed after current transition
            # Note: our implementation is inclusive: current transition can be sampled
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size

        # update info dictionaries
        info_dict_keys = self.infos[0, 0].keys()
        if self.copy_info_dict and "achieved_goal" in info_dict_keys and "desired_goal" in info_dict_keys:
            transition_info_dicts = self.infos[transition_indices, env_indices]
            for i in range(0, transition_info_dicts.shape[0]):
                batch_infos[i]["desired_goal"] = copy.deepcopy(transition_info_dicts[i]["achieved_goal"])

        return self.next_observations["achieved_goal"][transition_indices, env_indices], batch_infos
