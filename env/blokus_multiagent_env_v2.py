import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from ray.rllib.env import MultiAgentEnv

from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES
from game.move_generator import Move_generator
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS


class BlokusMultiAgentEnvV2(MultiAgentEnv):
    """Multi-agent Blokus environment with action masking and rewards."""

    def __init__(self, config=None):
        super().__init__()
        self.num_players = len(PLAYER_COLORS)
        self.agent_ids = [f"player_{i}" for i in range(self.num_players)]

        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(len(ALL_PIECES))
            for rot in range(4)
            for refl in range(2)
        ]
        self.skip_index = len(self.all_actions)
        self.all_actions.append(None)

        self.action_space = spaces.Discrete(len(self.all_actions))
        self.observation_space = spaces.Dict({
            "board": spaces.Box(-1, 1, (BOARD_SIZE, BOARD_SIZE), np.int8),
            "pieces_mask": spaces.MultiBinary(len(ALL_PIECES)),
        })

        self.game: Game | None = None
        self.current_agent_index: int = 0
        self.inactive_players: set[int] = set()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        self.inactive_players = set()
        self.current_agent_index = random.randrange(self.num_players)
        self.game.current_player_index = self.current_agent_index

        obs = {aid: self._compute_obs(i) for i, aid in enumerate(self.agent_ids)}
        return obs

    def step(self, action_dict):
        cur_idx = self.current_agent_index
        cur_id = self.agent_ids[cur_idx]
        action_idx = action_dict.get(cur_id, self.skip_index)

        reward, done = self._apply_action(cur_idx, action_idx)

        obs, rewards, dones, infos = {}, {}, {}, {}
        for i, aid in enumerate(self.agent_ids):
            obs[aid] = self._compute_obs(i)
            rewards[aid] = reward if i == cur_idx else 0.0
            dones[aid] = done
            infos[aid] = {"action_mask": self._compute_mask(i)}
        dones["__all__"] = done
        return obs, rewards, dones, infos

    def _apply_action(self, player_idx: int, action_idx: int):
        self.current_agent_index = player_idx
        self.game.current_player_index = player_idx
        player = self.game.players[player_idx]

        mask = self._compute_mask(player_idx)
        if action_idx == self.skip_index or not mask[action_idx]:
            remaining = sum(len(ALL_PIECES[i]) for i, avail in enumerate(player.pieces_mask) if avail)
            reward = -float(remaining)
            if remaining == 0:
                reward += 15.0
            if player_idx not in self.inactive_players:
                self.inactive_players.add(player_idx)
            else:
                reward = 0.0
            done = len(self.inactive_players) == self.num_players
            if not done:
                self._advance_to_next_active()
            return reward, done

        x, y, p_idx, rot, refl = self.all_actions[action_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(player)
        coords = next(c for vx, vy, pi, r, rf, c in raw_valid if (vx, vy, pi, r, rf) == (x, y, p_idx, rot, refl))
        self.game.board.place_piece(p_idx, coords, player)
        player.drop_piece(p_idx)

        reward = float(len(coords))
        done = False
        self._advance_to_next_active()
        return reward, done

    def _advance_to_next_active(self):
        for _ in range(self.num_players):
            self.game.next_turn()
            idx = self.game.current_player_index
            if idx not in self.inactive_players:
                self.current_agent_index = idx
                return

    def _compute_obs(self, player_idx: int):
        player = self.game.players[player_idx]
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.game.board.grid[r][c]
                if cell == player.color:
                    board[r, c] = 1
                elif cell is not None:
                    board[r, c] = -1
        return {"board": board, "pieces_mask": player.pieces_mask.copy()}

    def _compute_mask(self, player_idx: int) -> np.ndarray:
        player = self.game.players[player_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(player)
        valid_keys = {(vx, vy, p, r, rf) for vx, vy, p, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
        for idx, action in enumerate(self.all_actions):
            if action is None:
                if not valid_keys:
                    mask[idx] = True
            elif action in valid_keys:
                mask[idx] = True
        return mask

    def render(self, mode="human"):
        print(f"=== Current Player: player_{self.current_agent_index} ({self.game.players[self.current_agent_index].color}) ===")
        self.game.board.display()

    @property
    def observation_spaces(self):
        return {aid: self.observation_space for aid in self.agent_ids}

    @property
    def action_spaces(self):
        return {aid: self.action_space for aid in self.agent_ids}

