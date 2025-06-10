import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from env.blokus_multiagent_env_v2 import BlokusMultiAgentEnvV2


def env_creator(config=None):
    return BlokusMultiAgentEnvV2()


register_env("blokus_ma_v2", env_creator)


def main():
    ray.init(ignore_reinit_error=True)

    temp_env = env_creator()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    policies = {
        pid: (None, obs_space, act_space, {})
        for pid in temp_env.agent_ids
    }

    def policy_mapping_fn(agent_id, *_):
        return agent_id

    config = (
        PPOConfig()
        .environment("blokus_ma_v2")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(model={"fcnet_hiddens": [256, 256]})
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
    )

    algo = config.build()

    for i in range(2):
        result = algo.train()
        print(f"Iteration {i}: episode_reward_mean = {result['episode_reward_mean']}")

    checkpoint = algo.save("rllib_multiagent_v2")
    print("Checkpoint saved at", checkpoint)


if __name__ == "__main__":
    main()
