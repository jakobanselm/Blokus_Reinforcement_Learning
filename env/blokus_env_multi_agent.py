import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np
import logging

from ray.rllib.env import MultiAgentEnv

from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES
from game.move_generator import Move_generator
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS

logging.basicConfig(level=logging.INFO)

class BlokusMultiAgentEnv(MultiAgentEnv):
    """
    A RLlib-compatible MultiAgentEnv for Blokus self-play.
    Each agent_id is "player_0", "player_1", … and controls exactly one color.
    Turns advance round-robin; non-current agents submit dummy actions.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super().__init__()
        # Initialize random number generator for reproducibility
        self.np_random, _ = seeding.np_random(None)

        # Number of players and their agent IDs
        self.num_players = len(PLAYER_COLORS)
        self.agent_ids = [f"player_{i}" for i in range(self.num_players)]

        # Precompute full action list: placements and skip action
        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(len(ALL_PIECES))
            for rot in range(4)
            for refl in range(2)
        ]
        # Add skip action as last index
        self.skip_index = len(self.all_actions)
        self.all_actions.append(None)

        # Define shared action and observation spaces
        self.action_space = spaces.Discrete(len(self.all_actions))
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "pieces_mask": spaces.MultiBinary(len(ALL_PIECES)),
        })

        # Initialize environment state placeholders
        self.game: Game = None
        self.current_agent_index: int = 0
        self.inactive_players: set[int] = set()

        # Build mapping from action tuple to index for fast mask computation
        self._action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}

    def seed(self, seed=None):
        """
        Set the seed for random operations and return the seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the game to an initial state.
        Randomly select starting player using the seeded RNG.
        Returns initial observations dict for each agent.
        """
        self.inactive_players.clear()
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)

        # Choose starting player in a reproducible way
        self.current_agent_index = int(self.np_random.integers(self.num_players))
        self.game.current_player_index = self.current_agent_index

        # Compute and return observations for all agents
        return {
            agent_id: self._compute_obs(idx)
            for idx, agent_id in enumerate(self.agent_ids)
        }

    def step(self, action_dict):
        """
        Execute a step given dict of agent actions.
        Only the current agent's action is applied; others receive zero reward.
        Returns (obs, rewards, dones, infos).
        """
        cur_idx = self.current_agent_index
        cur_id = self.agent_ids[cur_idx]
        action_idx = action_dict[cur_id]

        # Validate provided action index
        self._validate_action_idx(action_idx)

        # Apply action for current agent
        reward, done, _ = self._apply_action(cur_idx, action_idx)

        # Build outputs for all agents
        obs, rewards, dones, infos = {}, {}, {}, {}
        for idx, agent_id in enumerate(self.agent_ids):
            obs[agent_id] = self._compute_obs(idx)
            rewards[agent_id] = reward if idx == cur_idx else 0.0
            dones[agent_id] = done
            infos[agent_id] = {"action_mask": self._compute_mask(idx)}

        dones["__all__"] = done
        return obs, rewards, dones, infos

    def _validate_action_idx(self, action_idx: int):
        """
        Validate that the action index is within the valid range [0, skip_index].
        Raises ValueError if invalid.
        """
        if not 0 <= action_idx <= self.skip_index:
            raise ValueError(
                f"Invalid action index {action_idx}; must be between 0 and {self.skip_index}."
            )

    def _apply_action(self, player_idx: int, action_idx: int):
        """
        Apply the specified action for the given player index.
        Returns (reward, done, info).
        """
        # Set current player in game state
        self.current_agent_index = player_idx
        self.game.current_player_index = player_idx
        current_player = self.game.players[player_idx]

        # Compute valid action mask for this player
        mask = self._compute_mask(player_idx)

        # If skip or invalid action, end player's game
        if action_idx == self.skip_index or not mask[action_idx]:
            # Compute end-of-game reward penalty or bonus
            remaining = sum(
                len(ALL_PIECES[i])
                for i, available in enumerate(current_player.pieces_mask) if available
            )
            reward = 15.0 if remaining == 0 else -float(remaining)

            # First time penalty only, subsequent skips yield zero
            if player_idx not in self.inactive_players:
                self.inactive_players.add(player_idx)
            else:
                reward = 0.0

            done = len(self.inactive_players) == self.num_players
            if not done:
                self._advance_to_next_active()
            return reward, done, {"action_mask": self._compute_mask(self.current_agent_index)}

        # Normal valid move
        x, y, p_idx, rot, refl = self.all_actions[action_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(current_player)
        coords = next(
            coord for vx, vy, pi, r, rf, coord in raw_valid
            if (vx, vy, pi, r, rf) == (x, y, p_idx, rot, refl)
        )
        self.game.board.place_piece(p_idx, coords, current_player)
        current_player.drop_piece(p_idx)

        reward, done = 0.0, False
        self._advance_to_next_active()
        return reward, done, {"action_mask": self._compute_mask(self.current_agent_index)}

    def _advance_to_next_active(self):
        """
        Advance turn in round-robin until an active player is found.
        """
        for _ in range(self.num_players):
            self.game.next_turn()
            idx = self.game.current_player_index
            if idx not in self.inactive_players:
                self.current_agent_index = idx
                return

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

    def _compute_mask(self, player_idx: int) -> np.ndarray:
        """
        Build action mask for the given player:
        - True for legal moves; skip action only if no legal moves
        """
        player = self.game.players[player_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(player)
        valid_actions = {(vx, vy, pi, r, rf) for vx, vy, pi, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
        for action in valid_actions:
            mask[self._action_to_index[action]] = True
        if not valid_actions:
            mask[self.skip_index] = True
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
