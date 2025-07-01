import subprocess
import sys


def check_dependencies():
    result = subprocess.run([sys.executable, "test_requirements.py"])
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    check_dependencies()

    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    from ray.tune.registry import register_env
    from env.blokus_env_multi_agent_ray_rllib import BlokusMultiAgentEnv

    def env_creator(config=None):
        env = BlokusMultiAgentEnv(config)
        if not hasattr(env, "observation_space"):
            env.observation_space = env.observation_spaces[env.possible_agents[0]]
        if not hasattr(env, "action_space"):
            env.action_space = env.action_spaces[env.possible_agents[0]]
        return env

    register_env("blokus_multi_agent", env_creator)

    dummy = env_creator()
    obs_space = dummy.observation_spaces[dummy.possible_agents[0]]
    act_space = dummy.action_spaces[dummy.possible_agents[0]]

    config = (
        PPOConfig()
        .environment("blokus_multi_agent")
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={
                "shared": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
            }
        )
        .training(train_batch_size=200)
    )

    algo = config.build()

    dummy.reset()
    for _ in range(2):
        algo.train()

    print("Sanity check passed")


if __name__ == "__main__":
    main()