import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env.blokus_env_multi_agent_ray_rllib import BlokusMultiAgentEnv

# 1. Umgebung registrieren
def env_creator(env_config):
    return BlokusMultiAgentEnv(env_config)

register_env("blokus_multi_agent", env_creator)

# 2. Ray initialisieren
ray.init(ignore_reinit_error=True)

# 3. Eine temporäre Umgebung erstellen, um die Spaces zu bekommen
temp_env = BlokusMultiAgentEnv()
obs_space = temp_env.observation_space
act_space = temp_env.action_space
agent_ids = temp_env.get_agent_ids()
temp_env.close()

# 4. PPO-Konfiguration für Multi-Agent-Training
config = (
    PPOConfig()
    .environment(
        "blokus_multi_agent",
        env_config={},
    )
    .framework("torch")
    .env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=1,
    )
    .multi_agent(
        policies={
            "shared_policy": (None, obs_space, act_space, {}),
        },
        policies_to_train=["shared_policy"],
    )
    .training(
        gamma=0.99,
        lr=5e-5,
        train_batch_size=4000,
        num_sgd_iter=10,
        model={
            "use_action_masking": True,
            "fcnet_hiddens": [256, 256],
        },
    )
    .resources(
        num_gpus=0,
    )
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# 5. Algorithmus erstellen
algo = config.build()

# 6. Training durchführen
for i in range(100):
    result = algo.train()
    reward_mean = result.get("episode_reward_mean", float('nan'))
    print(f"Iter: {i:03d}, Mean Reward: {reward_mean:.2f}")

    if (i + 1) % 10 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

# 7. Aufräumen
algo.stop()
ray.shutdown()