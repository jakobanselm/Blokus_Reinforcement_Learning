import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES
from game.move_generator import Move_generator
from global_constants import BOARD_SIZE, PLAYER_COLORS


class Blokus_Env_Masked(gym.Env):
    """
    Gym environment for Blokus with a flattened Discrete action space and
    action masking for invalid moves.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.all_pieces = ALL_PIECES
        self.num_pieces = len(self.all_pieces)

        # Create list of all possible actions: (x, y, piece_index, rotation, reflect_flag)
        self.all_actions = [
            (x, y, piece, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for piece in range(self.num_pieces)
            for rot in range(4)
            for refl in range(2)
        ]

        # Define a flat discrete action space over all action indices
        self.action_space = spaces.Discrete(len(self.all_actions))

        # Observation space includes the board state and available pieces mask
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            'pieces_mask': spaces.MultiBinary(self.num_pieces)
        })

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and initialize a new game.

        Returns:
            obs (dict): Initial observation
            info (dict): Empty info dict
        """
        # Seed randomness for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reinitialize the game instance
        self.game.__init__(board_size=BOARD_SIZE, player_colors=["X", "O"])
        self.current_player = self.game.players[0]

        observation = self._get_obs()
        return observation, {}

    def step(self, action_idx):
        """
        Execute an action by index, applying action masking.

        Args:
            action_idx (int): Index of the chosen action in self.all_actions

        Returns:
            observation (dict): Next observation
            reward (float): Reward earned by the action
            terminated (bool): Whether the game is over
            info (dict): Contains 'action_mask' for the next step
        """
        # Decode action index to actual move parameters
        x, y, piece, rotation, reflect = self.all_actions[action_idx]

        # Compute valid moves for the current player
        raw_valid = Move_generator(self.game.board).get_valid_moves(self.current_player)
        valid_keys = {(vx, vy, p, r, rf) for vx, vy, p, r, rf, _ in raw_valid}

        # Check if the selected action is valid
        if (x, y, piece, rotation, reflect) not in valid_keys:
            raise ValueError(f"Invalid action index: {action_idx}")

        # Retrieve placement coordinates and apply the move
        coords = next(
            coords for vx, vy, p, r, rf, coords in raw_valid
            if (vx, vy, p, r, rf) == (x, y, piece, rotation, reflect)
        )
        success = self.game.board.place_piece(piece, coords, self.current_player)
        if not success:
            raise RuntimeError("Board rejected a valid action.")

        # Calculate reward as number of cells placed
        reward = float(len(coords))

        # Update game state for next player
        self.current_player.drop_piece(piece)
        self.game.next_turn()
        self.current_player = self.game.players[self.game.current_player_index]

        # Determine if the game has ended
        terminated = not bool(Move_generator(self.game.board).get_valid_moves(self.current_player))

        # Build next observation and action mask
        observation = self._get_obs()
        info = {'action_mask': self.get_action_mask()}

        truncated = False
        return observation, reward, terminated, truncated, info

    def get_action_mask(self):
        """
        Generate a binary mask over all action indices, indicating valid moves.

        Returns:
            mask (np.ndarray): Boolean array of shape (num_actions,)
        """
        raw_valid = Move_generator(self.game.board).get_valid_moves(self.current_player)
        valid_keys = {(vx, vy, p, r, rf) for vx, vy, p, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
        for idx, action in enumerate(self.all_actions):
            if action in valid_keys:
                mask[idx] = True
        return mask

    def _get_obs(self):
        """
        Construct the observation dictionary containing the board matrix and pieces mask.

        Returns:
            obs (dict): {'board': np.ndarray, 'pieces_mask': np.ndarray}
        """
        # Encode board state: 1 for current player's cells, -1 for opponent's, 0 otherwise
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.game.board.grid[row][col]
                if cell == self.current_player.color:
                    board[row, col] = 1
                elif cell is not None:
                    board[row, col] = -1

        # Copy available pieces mask from current player
        pieces_mask = self.current_player.pieces_mask.copy()

        return {'board': board, 'pieces_mask': pieces_mask}

    def render(self, mode='human'):
        """
        Render the current board to the console.
        """
        self.game.board.display()
