from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from env.blokus_env_masked import  Blokus_Env_Masked
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS

env = Blokus_Env_Masked(game=Game(BOARD_SIZE, PLAYER_COLORS))
wrapped = ActionMasker(env, mask_fn=lambda e: e.get_action_mask())
model = PPO("MlpPolicy", wrapped, verbose=1, tensorboard_log="./tb_blokus/")
model.learn(total_timesteps=1_000)
model.save("ppo_blokus")