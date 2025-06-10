import numpy as np
from env.blokus_multiagent_env_v3 import BlokusMultiAgentEnvV3
from global_constants import BOARD_SIZE


def test_multiagent_env_v3_basic():
    env = BlokusMultiAgentEnvV3()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.agent_ids)

    for idx, aid in enumerate(env.agent_ids):
        o = obs[aid]
        assert o["board"].shape == (BOARD_SIZE, BOARD_SIZE)
        assert o["board"].dtype == np.int8
        assert o["pieces_mask"].dtype == np.int8
        mask = env._compute_mask(idx)
        assert mask.dtype == bool

    done = False
    steps = 0
    while not done and steps < 200:
        cur_idx = env.current_agent_index
        cur_id = env.agent_ids[cur_idx]
        mask = env._compute_mask(cur_idx)
        valid = np.flatnonzero(mask)
        action_dict = {aid: env.skip_index for aid in env.agent_ids}
        if valid.size:
            action_dict[cur_id] = int(np.random.choice(valid))
        obs, rewards, dones, infos = env.step(action_dict)
        assert set(rewards.keys()) == set(env.agent_ids)
        assert set(infos.keys()) == set(env.agent_ids)
        for aid in env.agent_ids:
            assert isinstance(rewards[aid], float)
            assert isinstance(dones[aid], bool)
            assert "action_mask" in infos[aid]
        done = dones["__all__"]
        steps += 1
    assert done
