import numpy as np
from env.blokus_env_multi_agent_ray_rllib import BlokusMultiAgentEnv
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def test_multiagent_env_smoke():
    env = BlokusMultiAgentEnv()

    # 1) Reset → only the starting agent has an obs/info
    obs_dict, info_dict = env.reset()
    assert isinstance(obs_dict, dict)
    assert len(obs_dict) == 1
    start_id = next(iter(obs_dict.keys()))
    assert start_id in env.possible_agents

    # Obs must contain 'board' and 'pieces_mask'
    obs = obs_dict[start_id]
    assert 'board' in obs and 'pieces_mask' in obs
    assert isinstance(obs['board'], np.ndarray)
    assert isinstance(obs['pieces_mask'], np.ndarray)

    # 2) Smoke run: up to 400 steps or until done
    for step in range(400):
        # Current acting agent
        cur_id = next(iter(obs_dict.keys()))
        mask = info_dict[cur_id]['action_mask']
        valid = np.flatnonzero(mask)
        # Choose a random valid action
        action = int(np.random.choice(valid)) if valid.size else env.skip_index
        action_dict = {cur_id: action}

        # 3) Step
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        cur_id = next(iter(rew_dict.keys()))
        if rew_dict[cur_id] != 0:
            logging.info(f"Step {step+1}: Agent {cur_id} received reward {rew_dict[cur_id]}")
        env.render(mode="human")

        # 4) Structure checks
        # a) Only one obs for next agent
        assert len(obs_dict) == 1
        next_id = next(iter(obs_dict.keys()))
        assert next_id in env.possible_agents

        # b) Reward only for previous agent
        assert set(rew_dict.keys()) == {cur_id}
        assert isinstance(rew_dict[cur_id], float)

        # c) Termination/truncation keys
        assert set(term_dict.keys()) == {"__all__"}
        assert isinstance(term_dict["__all__"], bool)
        assert set(trunc_dict.keys()) == {"__all__"}
        assert isinstance(trunc_dict["__all__"], bool)

        # d) Info only for next agent
        assert set(info_dict.keys()) == {next_id}
        mask2 = info_dict[next_id]['action_mask']
        assert isinstance(mask2, np.ndarray) and mask2.dtype == bool

        # e) End if done
        if term_dict["__all__"]:
            logging.info(f"Episode nach {step+1} Schritten beendet.")
            break
    else:
        raise AssertionError("Env hat nach 400 Steps nicht beendet")

    logging.info("MultiAgentEnv smoke test passed!")


def test_multiagent_env_smoke_with_render():
    env = BlokusMultiAgentEnv()
    obs_dict, info_dict = env.reset()  # Reset nur EINMAL am Anfang

    for step in range(100):
        # Aktueller Agent aus der letzten Observation
        cur_id = next(iter(obs_dict.keys()))  # <- Verwende obs_dict, nicht env.reset()!
        mask = info_dict[cur_id]['action_mask']
        valid = np.flatnonzero(mask)
        action = int(np.random.choice(valid)) if valid.size else env.skip_index

        # Action ausführen
        action_dict = {cur_id: action}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        # Render
        frame = env.render(mode="rgb_array")
        plt.figure(figsize=(4,4))
        plt.imshow(frame)
        plt.axis("off")
        plt.title(f"Step {step+1}")
        plt.show()

        if term_dict["__all__"]:
            break


if __name__ == "__main__":
    test_multiagent_env_smoke()
