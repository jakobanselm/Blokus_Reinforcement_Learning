import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pieces_definition import PIECES_DEFINITION as ALL_PIECES
from piece import Piece
from move_generator import Move_generator
import random

class Blokus_Env_Masked(gym.Env):
    """
    BlokusEnv with action masking: only legal moves are executed,
    invalid actions raise an error. Observation now includes 'valid_moves'.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.all_pieces = ALL_PIECES
        self.num_pieces = len(self.all_pieces)

        # Define observation and action spaces; agent must use the mask
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=1, shape=(14,14), dtype=np.int8),
            'pieces_mask': spaces.MultiBinary(self.num_pieces)
        })
        self.action_space = spaces.Dict({
            'x': spaces.Discrete(14),
            'y': spaces.Discrete(14),
            'piece': spaces.Discrete(self.num_pieces),
            'rotation': spaces.Discrete(4),
            'reflect': spaces.Discrete(2)
        })

    def reset(self, *, seed=None, options=None):
        # Seed RNGs for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize a new Game instance
        self.game.__init__(board_size=14, player_colors=["X", "O"])
        self.current_player = self.game.players[0]
        return self._get_obs()

    def step(self, action):
        """
        Executes a single action (must be in valid_moves).
        Returns (obs, reward, done, info).
        """
        # 1) Alle gültigen Moves inkl. coords holen
        raw_valid = Move_generator(self.game.board)\
                    .get_valid_moves(self.current_player)
        # 2) Mapping und Set bauen
        mapping = {
            (x, y, p_idx, rot, refl): coords
            for x, y, p_idx, rot, refl, coords in raw_valid
        }
        valid_set = set(mapping.keys())

        # 3) Key für die gewählte Action
        key = (
            action['x'], action['y'],
            action['piece'],
            action['rotation'], action['reflect']
        )
        if key not in valid_set:
            raise ValueError(f"Invalid action: {action}")

        # 4) coords direkt aus dem Mapping nehmen
        coords = mapping[key]

        # 5) Piece platzieren
        success = self.game.board.place_piece(
            action['piece'], coords, self.current_player
        )
        if not success:
            # zum Debug ggf.:
            self.game.board.display()
            print("coords:", coords)
            raise RuntimeError("Board rejected a valid move.")

        # 6) Reward & Turn-Handling
        reward = float(len(coords))
        self.current_player.drop_piece(action['piece'])
        self.game.next_turn()
        self.current_player = self.game.players[self.game.current_player_index]

        done = not Move_generator(self.game.board)\
                       .get_valid_moves(self.current_player)

        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        """Builds the observation dict with board, pieces_mask, valid_moves."""
        # board & mask
        board = np.zeros((14,14),dtype=np.int8)
        for y in range(14):
            for x in range(14):
                cell = self.game.board.grid[y][x]
                if cell == self.current_player.color:
                    board[y,x] = 1
                elif cell is not None:
                    board[y,x] = -1
        mask = self.current_player.pieces_mask.copy()

        # valid_moves list
        raw_valid = Move_generator(self.game.board)\
                    .get_valid_moves(self.current_player)
        valid_moves = [
            {'x': x, 'y': y, 'piece': p, 'rotation': rot, 'reflect': refl}
            for x,y,p,rot,refl,_ in raw_valid
        ]

        return {'board': board,
                'pieces_mask': mask,
                'valid_moves': valid_moves}

    def render(self, mode='human'):
        # Display the board via the underlying game
        self.game.board.display()
