
#RL libarys
from gymnasium import spaces
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

#Helper
import numpy as np
import logging
from typing import Dict, Tuple, Any, Set, Optional, List
logging.basicConfig(level=logging.INFO)

#Game logic
from game.move_generator import Move_generator
from game.game import Game

#Global constants
from global_constants import PLAYER_COLORS, BOARD_SIZE
from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES

class BlokusMultiAgentEnv(MultiAgentEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "name": "Blokus_v1"}


    ObsDict = Dict[str, Dict[str, np.ndarray]]
    RewardDict = Dict[str, float]
    TerminatedDict = Dict[str, bool]
    TruncatedDict = Dict[str, bool]
    InfoDict = Dict[str, Dict[str, Any]]
    StepReturn = Tuple[ObsDict, RewardDict, TerminatedDict, TruncatedDict, InfoDict]
    ResetReturn = Tuple[ObsDict, InfoDict]


    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(len(PLAYER_COLORS))]
        self.agents = self.possible_agents

        # Define shared action and observation spaces
        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(len(ALL_PIECES))
            for rot in range(4)
            for refl in range(2)
        ]
        self.all_actions.append(None)      # Länge N+1
        # skip_index ist jetzt genau der letzte Slot
        self.skip_index = len(self.all_actions) - 1  # == N
        # Discrete–Space deckt 0…N ab
        self.action_space = spaces.Discrete(len(self.all_actions))
        _action_space_all = spaces.Discrete(len(self.all_actions))
        _observation_space_all = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "pieces_mask": spaces.MultiBinary(len(ALL_PIECES))
        })
        self.action_spaces = {
            agent: _action_space_all
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: _observation_space_all
            for agent in self.possible_agents
        }

        # Initialize environment state placeholders
        self.game: Game = None
        self.inactive_players: set[int] = set()
        self.current_agent_index: int = 0
        self.num_players: int = len(PLAYER_COLORS)

        # Build mapping from action tuple to index for fast mask computation
        self._action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}


    def reset(self, *, seed=421, options=None) -> ResetReturn:
        """
        Reset the game to an initial state.
        Randomly select starting player using the seeded RNG.
        Returns initial observations dict for each agent.
        """
        self.np_random, _ = seeding.np_random(seed)
        self.inactive_players.clear()

        self.agents = self.possible_agents[:]
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        # Choose starting player in a reproducible way
        self.current_agent_index = int(self.np_random.integers(self.num_players))
        self.game.current_player_index = self.current_agent_index
        # Compute and return observations for all agents
        agent_id = self.possible_agents[self.current_agent_index]
        obs_dict  = {agent_id: self._compute_obs(self.current_agent_index)}
        info_dict = {agent_id: {"action_mask": self._compute_mask(self.current_agent_index)}}
        return obs_dict, info_dict

    def step(self, action_dict: Dict[str, int]) -> StepReturn:
        """
        Führt einen Zeitschritt in der Umgebung aus. Entspricht der Gymnasium-API.
        """
        cur_idx = self.current_agent_index
        cur_id = self.agents[cur_idx]
        action_idx = action_dict[cur_id]

        if not (0 <= action_idx < self.action_space.n):
            raise ValueError(f"Ungültiger Action-Index {action_idx}")

        action = action_dict[cur_id]
        reward, terminated = self._apply_action(cur_idx, action)
        truncated = False
        if not terminated:
            self._advance_to_next_active()
        next_idx = self.current_agent_index
        next_id  = self.possible_agents[next_idx]

        obs_dict     = { next_id: self._compute_obs(next_idx) }
        info_dict    = { next_id: { "action_mask": self._compute_mask(next_idx) } }
        reward_dict  = { cur_id: reward }
        term_dict    = { "__all__": terminated }
        trunc_dict   = { "__all__": truncated }

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict
            

    
    def _compute_obs(self, player_idx: int):
        """
        Build the observation for the given player:
        - 'board': 1 for own stones, -1 for opponents, 0 for empty
        - 'pieces_mask': Boolean mask of remaining pieces
        """
        player = self.game.players[player_idx]
        grid = np.array(self.game.board.grid, dtype=object)

        own_mask = (grid == player.color)
        opp_mask = (grid != None) & (~own_mask)

        board = own_mask.astype(np.int8) - opp_mask.astype(np.int8)
        return {"board": board, "pieces_mask": player.pieces_mask.copy()}
    
    def _apply_action(self, player_idx: int, action_idx: int) -> Tuple[float, bool]:
        """
        Wendet die Aktion an und gibt (reward, terminated) zurück.
        """
        self.current_agent_index = player_idx
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
            return reward, terminated

        # Gültiger Zug
        x, y, p_idx, rot, refl = self.all_actions[action_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(current_player)
        coords = next(
            coord for vx, vy, pi, r, rf, coord in raw_valid
            if (vx, vy, pi, r, rf) == (x, y, p_idx, rot, refl)
        )
        self.game.board.place_piece(p_idx, coords, current_player)
        current_player.drop_piece(p_idx)


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
        raw_valid = Move_generator(self.game.board).get_valid_moves(player)
        valid_actions = {(vx, vy, pi, r, rf) for vx, vy, pi, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
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
            self.game.board.display()
            logging.info(f"=== Current Player: player_{self.current_agent_index} ({self.game.players[self.current_agent_index].color}) ===")
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