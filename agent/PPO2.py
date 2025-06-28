# file: agent/ppo_ray_module.py

import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env

# Deine Umgebung
from env.blokus_env_multi_agent import BlokusMultiAgentEnv

if __name__ == "__main__":
    # 1) Starte Ray
    ray.init()

    # 2) Registriere Deine Umgebung mit dem Compatibility-Wrapper
    register_env(
        "blokus_multi_agent",
        lambda cfg: MultiAgentEnvCompatibility(BlokusMultiAgentEnv(cfg))
    )  # 

    # 3) Erstelle die PPOConfig im neuen API-Stack
    config = (
        PPOConfig()
        # a) Umgebung
        .environment(env="blokus_multi_agent")
        # b) Framework
        .framework("torch")
        # c) RLModule-Teil: DefaultPPOTorchRLModule + Catalog
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                catalog_class=PPOCatalog,
                model_config={"use_action_masking": True},
            )
        )
        # d) Sampling mit 1 EnvRunner (EnvVectorization=SYNC)
        .env_runners(num_env_runners=4)
        # e) Multi-Agent Setup (Parameter-Sharing)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
    )

    # 4) Baue und starte den Algorithmus
    algo = config.build()

    # 5) Training
    for i in range(10):
        result = algo.train()
        print(
            f"Iter {i+1:2d}  "
            f"reward_mean={result['episode_reward_mean']:.2f}  "
            f"len_mean={result['episode_len_mean']:.1f}"
        )

    # 6) Checkpoint
    checkpoint = algo.save()
    print(f"Checkpoint gespeichert in: {checkpoint}")

    # 7) Aufräumen
    algo.stop()
    ray.shutdown()
