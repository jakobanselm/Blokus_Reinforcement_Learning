from env.blokus_env_masked import Blokus_Env_Masked
from game.game import Game
import numpy as np
from numpy import random
from global_constants import BOARD_SIZE, PLAYER_COLORS


def test_blokus_ev_smoke():
    game = Game(board_size = BOARD_SIZE, player_colors=PLAYER_COLORS)


    # 2) Environment aufbauen
    env = Blokus_Env_Masked(game)

    # 2) Reset
    obs, info = env.reset()

    # 3) Random Rollout
    for _ in range(100):
        # a) gültige Aktionen ermitteln
        #    info['action_mask'] beim ersten Schritt noch leer, also fallback auf Methode
        mask = info.get('action_mask', env.get_action_mask())
        valid_indices = np.flatnonzero(mask)

        # b) zufällige Aktion wählen
        action = random.choice(valid_indices)

        # c) ausführen
        obs, reward, terminated, truncated, info = env.step(action)

        # d) rendern & ausgabe
        env.render()
        print(f"Reward: {reward:.1f}\n")

        if terminated:
            obs, info = env.reset()


test_blokus_ev_smoke()







