import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.blokus_env_masked import Blokus_Env_Masked
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS


class SelfPlayEnv(gym.Env):
    """Environment wrapper to facilitate self-play between two PPO agents."""

    def __init__(self, opponent=None):
        # Fresh game for each environment instance
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        base_env = Blokus_Env_Masked(self.game)
        self.env = ActionMasker(base_env, lambda e: e.get_action_mask())
        self.opponent = opponent

        # Expose base spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # If opponent starts, play its move first
        if self.env.game.current_player_index == 1 and self.opponent is not None:
            mask = info.get('action_mask', self.env.get_action_mask())
            action, _ = self.opponent.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, _, info = self.env.step(action)
            if terminated:
                return obs, info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and self.opponent is not None:
            mask = info.get('action_mask', self.env.get_action_mask())
            opp_action, _ = self.opponent.predict(obs, action_masks=mask, deterministic=True)
            obs, reward_opp, terminated, truncated, info = self.env.step(opp_action)
            reward -= reward_opp
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        self.env.render()


def main():
    """Train two PPO agents against each other in alternating fashion."""
    agent1 = MaskablePPO("MultiInputPolicy", SelfPlayEnv(), verbose=1)
    agent2 = MaskablePPO("MultiInputPolicy", SelfPlayEnv(), verbose=1)

    iterations = 5
    timesteps = 10_000

    for _ in range(iterations):
        # Train agent1 against current agent2
        env1 = SelfPlayEnv(opponent=agent2)
        agent1.set_env(env1)
        agent1.learn(total_timesteps=timesteps)

        # Train agent2 against updated agent1
        env2 = SelfPlayEnv(opponent=agent1)
        agent2.set_env(env2)
        agent2.learn(total_timesteps=timesteps)

    agent1.save("ppo_agent1")
    agent2.save("ppo_agent2")


if __name__ == "__main__":
    main()
