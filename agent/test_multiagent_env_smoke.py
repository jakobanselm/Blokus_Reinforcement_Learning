# agent/test_multiagent_env_smoke.py

import numpy as np
from env.blokus_env_multi_agent import BlokusMultiAgentEnv
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def test_multiagent_env_smoke():
    # 1) Env instanziieren
    env = BlokusMultiAgentEnv()
    env.seed(42)

    # 2) Reset → dict von obs pro Agent
    obs_dict = env.reset()
    info_dict = {
        aid: { "action_mask": env._compute_mask(idx) }
        for idx, aid in enumerate(env.agent_ids)
    }
    assert isinstance(obs_dict, dict), "reset() muss ein Dict zurückgeben"
    assert set(obs_dict.keys()) == set(env.agent_ids), "reset() muss für jeden Agenten eine Obs liefern"

    # jede Obs muss 'board' und 'pieces_mask' enthalten
    for aid, obs in obs_dict.items():
        assert 'board' in obs and 'pieces_mask' in obs, f"Obs von {aid} fehlt ein Key"
        assert isinstance(obs['board'], np.ndarray)
        assert isinstance(obs['pieces_mask'], np.ndarray)

    # 3) ein paar Steps durchlaufen
    for step in range(400):
    # Bestimme aktuell handelnden Spieler und sein Mask
        cur_idx = env.current_agent_index
        cur_id  = env.agent_ids[cur_idx]
        mask    = info_dict[cur_id]['action_mask']
        valid   = np.flatnonzero(mask)

        # baue action_dict: alle gehen skip, nur current wählt aus valid
        action_dict = {aid: env.skip_index for aid in env.agent_ids}
        if valid.size:
            action_dict[cur_id] = int(np.random.choice(valid))

        obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

        env.render()

        # 5) Struktur-Checks
        # a) Obs/Reward/Done/Info keys
        assert set(obs_dict.keys()) == set(env.agent_ids)
        assert set(rew_dict.keys()) == set(env.agent_ids)
        # done muss zusätzlich "__all__" enthalten
        assert set(done_dict.keys()) == set(env.agent_ids) | {"__all__"}
        assert set(info_dict.keys()) == set(env.agent_ids)

        # b) Typen und Inhalte
        for aid in env.agent_ids:
            assert isinstance(rew_dict[aid], float), f"Reward von {aid} muss float sein"
            assert isinstance(done_dict[aid], bool),  f"done von {aid} muss bool sein"
            assert 'action_mask' in info_dict[aid],   f"info von {aid} braucht 'action_mask'"
            mask = info_dict[aid]['action_mask']
            assert isinstance(mask, np.ndarray) and mask.dtype == bool

        # c) abbrechen wenn alle fertig
        if done_dict["__all__"]:
            logging.info(f"Episode nach {step+1} Schritten beendet.")
            env.reset()
            info_dict = {
                 aid: {"action_mask": env._compute_mask(idx)}
                 for idx, aid in enumerate(env.agent_ids)
             }
            break
    else:
        raise AssertionError("Env hat nach 400 Steps nicht beendet")

    logging.info("MultiAgentEnv smoke test passed!")

import matplotlib.pyplot as plt

def test_multiagent_env_smoke_with_render():
    env = BlokusMultiAgentEnv()
    env.seed(42)
    obs = env.reset()

    # do a couple of random steps
    for step in range(100):
        # pick a random valid action for the current agent
        cur = env.current_agent_index
        mask = env._compute_mask(cur)
        valid = np.flatnonzero(mask)
        action = int(np.random.choice(valid)) if valid.size else env.skip_index

        obs, rew, done, info = env.step({aid: (action if aid == env.agent_ids[cur] else env.skip_index)
                                         for aid in env.agent_ids})

        # render and display
        frame = env.render(mode="rgb_array")   # returns H×W×3 uint8
        plt.figure(figsize=(4,4))
        plt.imshow(frame)
        plt.axis("off")
        plt.title(f"Step {step+1}")
        plt.show()

        if done["__all__"]:
            break


if __name__ == "__main__":
    test_multiagent_env_smoke()
    test_multiagent_env_smoke_with_render()
