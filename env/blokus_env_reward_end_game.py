import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES
from game.move_generator import Move_generator
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS


class Blokus_Env_Masked(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, game: Game):
        super().__init__()
        # track which players have dropped out
        self.inactive_players = set()

        self.game = game
        self.all_pieces = ALL_PIECES
        self.num_pieces = len(self.all_pieces)
        self.num_players = len(PLAYER_COLORS)

        # build full action list once
        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(self.num_pieces)
            for rot in range(4)
            for refl in range(2)
        ]
        # 1) add a special “skip” action at the end for players with no valid move
        self.skip_index = len(self.all_actions)    # index of the no-op action
        self.all_actions.append(None)              # None denotes “no move / skip”

        # now define the action space over the extended list
        self.action_space = spaces.Discrete(len(self.all_actions))

        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            'pieces_mask': spaces.MultiBinary(self.num_pieces)
        })

    def reset(self, *, seed=None, options=None):
        # English comment: reset inactive state and re-create game
        self.inactive_players = set()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # re-instantiate Game so everything is clean
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        # pick a random starting player
        self.game.current_player_index = random.randrange(self.num_players)
        self.current_player = self.game.players[self.game.current_player_index]

        obs = self._get_obs()
        return obs, {'action_mask': self.get_action_mask()}

    def step(self, action_idx: int):
        mask = self.get_action_mask()

        # decode action
        action = self.all_actions[action_idx]

        # 1) No-Op branch: skip/dropout
        if action is None:
            # compute end-reward for this player
            remaining = sum(len(ALL_PIECES[i])
                            for i, av in enumerate(self.current_player.pieces_mask) if av)
            reward = 15.0 if remaining == 0 else -1.0 * remaining
            # mark dropout und evtl. beenden
            self.inactive_players.add(self.game.current_player_index)
            print(f"Player {self.current_player.color} has no valid moves left. Reward: {reward:.1f}")
            print(f"Inactive players: {self.inactive_players}")
            print(f"Current player: {self.current_player.color} (index {self.game.current_player_index})")
            print(f"Player pieces mask: {self.current_player.pieces_mask}")

            done = (len(self.inactive_players) == self.num_players)
            if not done:
                self._advance_to_next_active()

            return self._get_obs(), reward, done, False, {'action_mask': self.get_action_mask()}

        # 2) normaler Zug
        x, y, p_idx, rot, refl = action
        raw_valid = Move_generator(self.game.board).get_valid_moves(self.current_player)
        coords = next(c for vx, vy, pi, r, rf, c in raw_valid
                      if (vx, vy, pi, r, rf) == (x, y, p_idx, rot, refl))

        self.game.board.place_piece(p_idx, coords, self.current_player)
        self.current_player.drop_piece(p_idx)

        # kein Zwischenschritt-Reward
        reward = 0.0
        self._advance_to_next_active()

        return self._get_obs(), reward, False, False, {'action_mask': self.get_action_mask()}


    def _advance_to_next_active(self):
        # English comment: rotate until finding player not in inactive_players
        for _ in range(self.num_players):
            self.game.next_turn()
            idx = self.game.current_player_index
            if idx not in self.inactive_players:
                self.current_player = self.game.players[idx]
                return

    def get_action_mask(self) -> np.ndarray:
        raw_valid = Move_generator(self.game.board).get_valid_moves(self.current_player)
        valid_keys = {(vx, vy, p, r, rf) for vx, vy, p, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
        for idx, action in enumerate(self.all_actions):
            if action is None:
                # only allow skip if no real moves exist
                if not valid_keys:
                    mask[idx] = True
            else:
                if action in valid_keys:
                    mask[idx] = True
        return mask


    def _get_obs(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.game.board.grid[row][col]
                if cell == self.current_player.color:
                    board[row, col] = 1
                elif cell is not None:
                    board[row, col] = -1
        pieces_mask = self.current_player.pieces_mask.copy()
        return {'board': board, 'pieces_mask': pieces_mask}

    def render(self, mode='human'):
        self.game.board.display()
