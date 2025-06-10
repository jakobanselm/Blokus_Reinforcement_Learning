from env.blokus_env_reward_end_game import Blokus_Env_Masked
from game.game import Game
import numpy as np
from numpy import random
from global_constants import BOARD_SIZE, PLAYER_COLORS


def test_blokus_env_smoke():
    # 1) Spiel-Objekt und Env erzeugen
    game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
    env = Blokus_Env_Masked(game)

    # 2) Reset und erstes Mask holen
    obs, info = env.reset()
    assert 'action_mask' in info, "Reset muss initial eine action_mask liefern"
    mask = info['action_mask']
    assert mask.any(), "Beim Reset müssen gültige Aktionen vorhanden sein"

    # 3) Zufälliger Rollout über alle Spieler bis Game Over
    done = False
    step_count = 0
    max_steps = 1000  # Notausstieg, falls was hängen bleibt
    while step_count < max_steps:
        # a) gültige Action-Indizes ermitteln
        mask = info.get('action_mask', env.get_action_mask())
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            print(f"Kein legaler Zug für Spieler {env.current_player.color}")

        current_player = env.current_player.color
        # b) zufällige Auswahl
        action = random.choice(valid_indices)

        print(f"Spieler {current_player} wählt Aktion: {action} (Index lengths {len(valid_indices)})")

        # c) Step ausführen
        obs, reward, done, truncated, info = env.step(int(action))
        if reward != 0:
            print(f"Spieler {current_player} hat {reward:.1f} Punkte erhalten")

        # d) ein paar einfache Checks
        assert isinstance(reward, float), "Reward muss float sein"
        assert isinstance(done, bool), "done muss bool sein"
        assert isinstance(truncated, bool), "truncated muss bool sein"


        env.render()
        step_count += 1
        if done:
            obs, info = env.reset()
           

        

    


if __name__ == "__main__":
    test_blokus_env_smoke()
    print("Smoke Test passed!")
