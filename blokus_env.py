import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pieces_definition import PIECES_DEFINITION as ALL_PIECES
from piece import Piece
from random import choice
from move_generator import Move_generator

# Reward shaping constants
REWARD_PIECE_NOT_AVAILABLE    = -1
REWARD_NO_POSSIBLE_MOVES      = -10
REWARD_NO_SUCCESSFUL_MOVES    = 0
REWARD_NOT_THE_RIGHT_ORIGIN   = -10

class BlokusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.move_gen = None
        self.all_pieces = ALL_PIECES
        self.num_pieces = len(self.all_pieces)

        # Define observation and action spaces
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
        self.current_player = None

    def reset(self):
        self.game.__init__(board_size=14, player_colors=["R", "B"])
        self.current_player = self.game.players[0]
        self.current_player.reset_pieces()
        return self._get_obs()

    def step(self, action):
        # Initialize
        done = False
        reward = 0.0
        ox, oy = action['x'], action['y']
        p_idx = action['piece']
        rot = action['rotation']
        refl = action['reflect']

        self.move_gen = Move_generator(self.game.board)

        # Gather valid moves/origins
        valid_origins = self.move_gen.get_valid_origins(self.current_player)
        valid_moves   = self.move_gen.get_valid_moves(self.current_player)


        # 1) Piece availability
        if self.current_player.pieces_mask[p_idx] == 0:
            # Penalty and fallback
            reward = REWARD_PIECE_NOT_AVAILABLE
            if valid_moves:
                X, Y, fb_idx, rotations, reflection, fb_coords = choice(valid_moves)
                self.game.board.place_piece(fb_idx, fb_coords, self.current_player)
                self.current_player.drop_piece(fb_idx)
                reward += REWARD_NO_SUCCESSFUL_MOVES
            else:
                done = True
            # Advance and return
            self.game.next_turn()
            self.current_player = self.game.players[self.game.current_player_index]
            return self._get_obs(), reward, done, {}

        # 2) Origin validity
        if (ox, oy) not in valid_origins:
            reward = REWARD_NOT_THE_RIGHT_ORIGIN
            if valid_moves:
                X, Y, fb_idx, rotations, reflection, fb_coords = choice(valid_moves)
                self.game.board.place_piece(fb_idx, fb_coords, self.current_player)
                self.current_player.drop_piece(fb_idx)
                reward += REWARD_NO_SUCCESSFUL_MOVES
            else:
                done = True
            self.game.next_turn()
            self.current_player = self.game.players[self.game.current_player_index]
            return self._get_obs(), reward, done, {}

        # 3) No legal moves at all
        if not valid_moves:
            return self._get_obs(), REWARD_NO_POSSIBLE_MOVES, True, {}

        # 4) Intended placement
        piece = Piece(ALL_PIECES[p_idx])
        coords = piece.get_positions((ox, oy), rotation=rot, reflect=refl)

        # 5) Try place
        if self.game.board.place_piece(p_idx, coords, self.current_player):
            reward = len(coords)
        else:
            reward = REWARD_NO_SUCCESSFUL_MOVES
            X, Y, fb_idx, rotations, reflection, fb_coords = choice(valid_moves)
            self.game.board.place_piece(fb_idx, fb_coords, self.current_player)
            self.current_player.drop_piece(fb_idx)
            # reward remains as fallback penalty

        # Advance turn
        self.game.next_turn()
        self.current_player = self.game.players[self.game.current_player_index]

        # 6) End-of-game bonus
        next_origins = self.move_gen.get_valid_origins(self.current_player)
        if not next_origins or self.current_player.pieces_mask.sum() == 0:
            done = True
            agent_mask = self.current_player.pieces_mask
            opp_player = self.game.players[1 - self.game.current_player_index]
            opp_mask = opp_player.pieces_mask
            agent_remain = sum(len(self.all_pieces[i]) for i, v in enumerate(agent_mask) if v)
            opp_remain   = sum(len(self.all_pieces[i]) for i, v in enumerate(opp_mask) if v)
            reward += float(opp_remain - agent_remain)

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        board = np.zeros((14,14), dtype=np.int8)
        for y in range(14):
            for x in range(14):
                cell = self.game.board.grid[y][x]
                if cell == self.current_player.color:
                    board[y, x] = 1
                elif cell is not None:
                    board[y, x] = -1
        mask = self.current_player.pieces_mask
        return {'board': board, 'pieces_mask': mask.copy()}

    def render(self, mode='human'):
        self.game.board.display()
