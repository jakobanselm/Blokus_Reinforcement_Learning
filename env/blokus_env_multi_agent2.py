import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np
import logging
from typing import Dict, Tuple, Any, Set, Optional, List

from ray.rllib.env import MultiAgentEnv

logging.basicConfig(level=logging.INFO)

class BlokusMultiAgentEnv(MultiAgentEnv):
    """
    Eine RLlib-kompatible MultiAgentEnv für Blokus, die den modernen Gymnasium- und Ray-APIs entspricht.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "Blokus_v1"}

    # Die Typ-Annotationen für die Rückgabewerte von step() werden hier zur besseren Lesbarkeit definiert
    ObsDict = Dict[str, Dict[str, np.ndarray]]
    RewardDict = Dict[str, float]
    TerminatedDict = Dict[str, bool]
    TruncatedDict = Dict[str, bool]
    InfoDict = Dict[str, Dict[str, Any]]
    StepReturn = Tuple[ObsDict, RewardDict, TerminatedDict, TruncatedDict, InfoDict]
    ResetReturn = Tuple[ObsDict, InfoDict]


    def __init__(self, config=None):
        super().__init__()
        # Das Seeding wird jetzt ausschließlich über reset(seed=...) gehandhabt.
        # Der np_random Generator wird beim ersten Aufruf von reset() initialisiert.
        self.np_random = None

        self.num_players = len(PLAYER_COLORS)
        
        # RLlib > 2.0 erfordert, dass `possible_agents` und `agents` definiert sind.
        # possible_agents: Alle Agenten, die jemals in der Umgebung existieren könnten.
        # agents: Die Agenten, die in der aktuellen Episode aktiv sind.
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]

        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(len(ALL_PIECES))
            for rot in range(4)
            for refl in range(2)
        ]
        self.skip_index = len(self.all_actions)
        self.all_actions.append(None) # None repräsentiert den "skip" Zug

        # Geteilte Action- und Observation-Spaces für alle Agenten
        self.action_space = spaces.Discrete(len(self.all_actions))
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8), # Empfehlung: Action Mask in die Obs
            "pieces_mask": spaces.MultiBinary(len(ALL_PIECES)),
        })
        
        self.game: Game = None
        self.current_agent_index: int = 0
        self.inactive_players: Set[int] = set()
        self._action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}

    # Die separate `seed`-Methode ist veraltet und sollte entfernt werden.
    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> ResetReturn:
        """
        Setzt die Umgebung zurück. Entspricht der Gymnasium-API.
        """
        # `super().reset()` kümmert sich um das Seeding, inkl. der Initialisierung von `self.np_random`
        super().reset(seed=seed)
        
        self.inactive_players.clear()
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        
        # `self.np_random` wird durch `super().reset()` initialisiert.
        self.current_agent_index = int(self.np_random.integers(self.num_players))
        self.game.current_player_index = self.current_agent_index

        # Setzt die Liste der aktiven Agenten für die neue Episode zurück
        self.agents = self.possible_agents[:]

        observations = {agent_id: self._compute_obs(idx) for idx, agent_id in enumerate(self.agents)}
        infos = {agent_id: {} for agent_id in self.agents} # Info-Dict kann anfangs leer sein

        return observations, infos

    def step(self, action_dict: Dict[str, int]) -> StepReturn:
        """
        Führt einen Zeitschritt in der Umgebung aus. Entspricht der Gymnasium-API.
        """
        cur_idx = self.current_agent_index
        cur_id = self.agents[cur_idx]
        action_idx = action_dict[cur_id]

        if not (0 <= action_idx <= self.skip_index):
            raise ValueError(f"Ungültiger Action-Index {action_idx}")

        reward, terminated = self._apply_action(cur_idx, action_idx)

        # Truncated ist False, da das Spiel nur durch Endbedingungen terminiert wird,
        # nicht durch ein künstliches Zeitlimit.
        truncated = False
        
        # Erstelle die Rückgabedictionaries für alle Agenten
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        
        # Der __all__-Schlüssel ist entscheidend, um RLlib das Ende der Episode zu signalisieren.
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        for idx, agent_id in enumerate(self.agents):
            obs[agent_id] = self._compute_obs(idx)
            rewards[agent_id] = reward if idx == cur_idx else 0.0
            terminateds[agent_id] = terminated
            truncateds[agent_id] = truncated
            infos[agent_id] = {} # Infos werden nach Bedarf gefüllt

        return obs, rewards, terminateds, truncateds, infos

    def _apply_action(self, player_idx: int, action_idx: int) -> Tuple[float, bool]:
        """
        Wendet die Aktion an und gibt (reward, terminated) zurück.
        """
        self.game.current_player_index = player_idx
        current_player = self.game.players[player_idx]
        
        # Die Action Mask wurde bereits in der Observation des vorherigen Schrittes berechnet
        valid_mask = self._compute_mask(player_idx)

        # Wenn der Zug ungültig ist oder gepasst wird
        if action_idx == self.skip_index or not valid_mask[action_idx]:
            reward = 0.0
            # Nur beim ersten Mal passen gibt es eine Strafe
            if player_idx not in self.inactive_players:
                remaining = sum(
                    len(ALL_PIECES[i])
                    for i, available in enumerate(current_player.pieces_mask) if available
                )
                reward = 15.0 if remaining == 0 else -float(remaining)
                self.inactive_players.add(player_idx)

            terminated = len(self.inactive_players) == self.num_players
            if not terminated:
                self._advance_to_next_active()
            return reward, terminated

        # Gültiger Zug
        x, y, p_idx, rot, refl = self.all_actions[action_idx]
        
        # Diese Logik hängt stark von Ihren Spiel-Klassen ab.
        # Annahme: place_piece und drop_piece modifizieren den Zustand korrekt.
        # raw_valid = MockMoveGenerator(self.game.board).get_valid_moves(current_player)
        # coords = next(...)
        # self.game.board.place_piece(...)
        current_player.drop_piece(p_idx)

        self._advance_to_next_active()
        return 0.0, False

    def _advance_to_next_active(self):
        """
        Wechselt zum nächsten Spieler, der noch aktiv ist.
        """
        for _ in range(self.num_players):
            self.game.next_turn()
            idx = self.game.current_player_index
            if idx not in self.inactive_players:
                self.current_agent_index = idx
                return

    def _compute_obs(self, player_idx: int) -> Dict[str, np.ndarray]:
        """
        Erstellt die Observation für einen Spieler.
        Empfehlung: Die Action Mask direkt in die Observation zu integrieren,
        da sie für die Entscheidung des Agenten kritisch ist.
        """
        player = self.game.players[player_idx]
        
        # Board-Repräsentation (1 für eigene, -1 für gegnerische Steine)
        grid = np.array(self.game.board.grid, dtype=object)
        own_mask = (grid == player.color)
        opp_mask = (grid != None) & (~own_mask)
        board_state = own_mask.astype(np.int8) - opp_mask.astype(np.int8)

        # Action Mask
        action_mask = self._compute_mask(player_idx)

        return {
            "board": board_state,
            "pieces_mask": player.pieces_mask.copy().astype(np.int8),
            "action_mask": action_mask.astype(np.int8)
        }

    def _compute_mask(self, player_idx: int) -> np.ndarray:
        """
        Erstellt die Action Mask für einen Spieler.
        """
        # Falls der Spieler inaktiv ist, kann er nur passen.
        if player_idx in self.inactive_players:
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[self.skip_index] = True
            return mask
            
        player = self.game.players[player_idx]
        raw_valid = MockMoveGenerator(self.game.board).get_valid_moves(player)
        valid_actions = {(vx, vy, pi, r, rf) for vx, vy, pi, r, rf, _ in raw_valid}

        mask = np.zeros(self.action_space.n, dtype=bool)
        if not valid_actions:
            mask[self.skip_index] = True
        else:
            for action in valid_actions:
                mask[self._action_to_index[action]] = True
        return mask

 
    def render(self, mode="human"):
        """
        Render the board.
        Modes:
         - 'human': log to console with ASCII
         - 'rgb_array': return an RGB image as numpy array
        """
        if mode == "human":
            logging.info(f"=== Current Player: player_{self.current_agent_index} ({self.game.players[self.current_agent_index].color}) ===")
            self.game.board.display()
        elif mode == "rgb_array":
            return self._render_frame()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render_frame(self) -> np.ndarray:
        """
        Build and return an RGB image of the current board state.
        Returns:
            np.ndarray of shape (BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8
        """
        grid = self.game.board.grid
        height, width = BOARD_SIZE, BOARD_SIZE
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Define a simple palette: empty white, then red, green, blue, yellow
        palette = [
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0)
        ]
        color_map = {None: palette[0]}
        for i, color in enumerate(PLAYER_COLORS):
            color_map[color] = palette[i+1]
        for r in range(height):
            for c in range(width):
                cell = grid[r][c]
                img[r, c] = color_map.get(cell, palette[0])
        return img

    def close(self):
        """
        Clean up any resources (placeholder).
        """
        pass

    @property
    def observation_spaces(self):
        """
        Return dict of observation spaces for each agent.
        """
        return {agent_id: self.observation_space for agent_id in self.agent_ids}

    @property
    def action_spaces(self):
        """
        Return dict of action spaces for each agent.
        """
        return {agent_id: self.action_space for agent_id in self.agent_ids}