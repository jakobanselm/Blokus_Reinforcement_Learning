from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.blokus_env_multi_agent import BlokusMultiAgentEnv
import ray

# Register your environment with a string name
def env_creator(env_config):
    return BlokusMultiAgentEnv(env_config)

register_env("blokus_ma_env", env_creator)

# Create an instance of the environment to get observation_space and action_space
temp_env = BlokusMultiAgentEnv()

config = (
    PPOConfig()
    .environment("blokus_ma_env")
    .framework("torch")
    .training(
        gamma=0.99,
        lr=0.0001,
        train_batch_size=2048,
        num_epochs=30,
    )
    .resources(
        num_gpus=0,
    )
    .multi_agent(
        policies={
            "blokus_policy": (None, temp_env.observation_space, temp_env.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "blokus_policy",
    )
    .rl_module(
        model_config={
            "use_action_masking": True,
        }
    )
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=1,
        rollout_fragment_length=512, # Dies wurde von 200 auf 512 geändert, um den Fehler zu beheben
        # Alternativ könntest du auch 'auto' verwenden: rollout_fragment_length='auto',
    )
)

# Initialize Ray
ray.init()

# Use build_algo() instead of build()
algo = config.build_algo()

# Training loop
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")

algo.stop()
ray.shutdown()