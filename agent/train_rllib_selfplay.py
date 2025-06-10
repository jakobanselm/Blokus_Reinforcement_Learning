import ray
from ray.rllib.algorithms.ppo import PPOConfig

from env.blokus_env_multi_agent import BlokusMultiAgentEnv


def main():
    """Train four PPO policies in self‑play using RLlib."""
    # Instantiate once to obtain spaces and agent ids
    temp_env = BlokusMultiAgentEnv()
    policy_ids = temp_env.agent_ids
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    policies = {
        pid: (None, obs_space, act_space, {})
        for pid in policy_ids
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        # Each agent uses its own policy
        return agent_id

    config = (
        PPOConfig()
        .environment(BlokusMultiAgentEnv)
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policy_ids),
        )
    )

    algo = config.build()

    for i in range(10):
        result = algo.train()
        print(f"Iteration {i}: episode_reward_mean = {result['episode_reward_mean']}")

    algo.save("rllib_blokus_agents")


if __name__ == "__main__":
    main()
