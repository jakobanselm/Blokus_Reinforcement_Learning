import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from ray.rllib.env import MultiAgentEnv

from game.pieces_definition import PIECES_DEFINITION as ALL_PIECES
from game.move_generator import Move_generator
from game.game import Game
from global_constants import BOARD_SIZE, PLAYER_COLORS


class BlokusMultiAgentEnv(MultiAgentEnv):
    """
    A RLlib‐compatible MultiAgentEnv for Blokus self-play.
    Each agent_id is “player_0”, “player_1”, … and controls exactly one color.
    Turns advance round-robin; non-current agents submit dummy actions.
    """

    def __init__(self, config=None):
        super().__init__()  
        # number of players / policies
        self.num_players = len(PLAYER_COLORS)
        self.agent_ids = [f"player_{i}" for i in range(self.num_players)]

        # build full action list once
        self.all_actions = [
            (x, y, p_idx, rot, refl)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for p_idx in range(len(ALL_PIECES))
            for rot in range(4)
            for refl in range(2)
        ]
        # add “skip” action at end
        self.skip_index = len(self.all_actions)
        self.all_actions.append(None)

        # all agents share same spaces
        self.action_space = spaces.Discrete(len(self.all_actions))
        self.observation_space = spaces.Dict({
            "board": spaces.Box(-1, 1, (BOARD_SIZE, BOARD_SIZE), np.int8),
            "pieces_mask": spaces.MultiBinary(len(ALL_PIECES)),
        })

        # Will be initialized in reset()
        self.game: Game = None
        self.current_agent_index: int = 0
        self.inactive_players: set[int] = set()

    def reset(self):
        """Reset board, pick random starting agent, clear inactive set."""
        self.inactive_players = set()
        self.game = Game(board_size=BOARD_SIZE, player_colors=PLAYER_COLORS)
        # pick random starter
        self.current_agent_index = random.randrange(self.num_players)
        self.game.current_player_index = self.current_agent_index

        # build initial obs for every agent
        obs = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs[agent_id] = self._compute_obs(i)
        return obs

    def step(self, action_dict):
        """
        action_dict: { agent_id: action_index, … }
        We only execute the action for the current agent; others get zero‐reward.
        """
        # identify current agent
        cur_idx = self.current_agent_index
        cur_id = self.agent_ids[cur_idx]
        action_idx = action_dict[cur_id]

        # apply current agent's action
        reward, done, info = self._apply_action(cur_idx, action_idx)

        # build outputs for all agents
        obs, rewards, dones, infos = {}, {}, {}, {}
        for i, agent_id in enumerate(self.agent_ids):
            obs[agent_id] = self._compute_obs(i)
            rewards[agent_id] = reward if i == cur_idx else 0.0
            dones[agent_id] = done
            
            # statt nur für den aktiven Spieler jetzt für alle:
            infos[agent_id] = {
                "action_mask": self._compute_mask(i)
            }

        # RLlib‐Flag für alle beendet
        dones["__all__"] = done
        return obs, rewards, dones, infos

    def _apply_action(self, player_idx: int, action_idx: int):
        """
        Execute one step for the given player index.
        Returns (reward, done, info).
        """
        # set game to that player's turn
        self.current_agent_index = player_idx
        self.game.current_player_index = player_idx
        self.current_player = self.game.players[player_idx]

        mask = self._compute_mask(player_idx)

        # 1) skip/dropout wenn Skip-Index oder die gewählte Aktion nicht im Masken-Array steht
        if action_idx == self.skip_index or not mask[action_idx]:
            # compute end‐reward
            remaining = sum(
                len(ALL_PIECES[i])
                for i, avail in enumerate(self.current_player.pieces_mask) if avail
            )
            reward = 15.0 if remaining == 0 else -float(remaining)

            # mark player inactive (only first time penalize)
            if player_idx not in self.inactive_players:
                self.inactive_players.add(player_idx)
            else:
                reward = 0.0  # no double‐punishment

            done = len(self.inactive_players) == self.num_players
            # advance turn if not all finished
            if not done:
                self._advance_to_next_active()
            return reward, done, {"action_mask": self._compute_mask(self.current_agent_index)}

        # 2) normal move
        action = self.all_actions[action_idx]
        x, y, p_idx, rot, refl = action
        raw_valid = Move_generator(self.game.board).get_valid_moves(self.current_player)
        coords = next(
            c for vx, vy, pi, r, rf, c in raw_valid
            if (vx, vy, pi, r, rf) == (x, y, p_idx, rot, refl)
        )
        self.game.board.place_piece(p_idx, coords, self.current_player)
        self.current_player.drop_piece(p_idx)

        # no intermediate reward
        reward = 0.0
        done = False

        # advance to next active player
        self._advance_to_next_active()
        return reward, done, {"action_mask": self._compute_mask(self.current_agent_index)}

    def _advance_to_next_active(self):
        """Round‐robin advance until finding a player not in inactive_players."""
        for _ in range(self.num_players):
            self.game.next_turn()
            idx = self.game.current_player_index
            if idx not in self.inactive_players:
                self.current_agent_index = idx
                return

    def _compute_obs(self, player_idx: int):
        """
        Build the observation for the given player:
         - board: 1 for own stones, -1 for any opponent, 0 empty
         - pieces_mask: which of this player's pieces remain
        """
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
        """
        Build action mask for a specific player:
         - legal moves → True
         - skip action → True only if no legal moves
        """
        player = self.game.players[player_idx]
        raw_valid = Move_generator(self.game.board).get_valid_moves(player)
        valid_keys = {(vx, vy, p, r, rf) for vx, vy, p, r, rf, _ in raw_valid}

        mask = np.zeros(len(self.all_actions), dtype=bool)
        for idx, action in enumerate(self.all_actions):
            if action is None:
                if not valid_keys:
                    mask[idx] = True
            else:
                if action in valid_keys:
                    mask[idx] = True
        return mask

    def render(self, mode="human"):

        """
        Druckt das Board und zeigt, welcher Agent gerade am Zug ist.
        """
        print(f"=== Current Player: player_{self.current_agent_index} ({self.game.players[self.current_agent_index].color}) ===")
        self.game.board.display()

    @property
    def observation_spaces(self):
        # keys müssen exakt den agent_ids entsprechen, z.B. "player_0", "player_1", …
        return {
            agent_id: self.observation_space
            for agent_id in self.agent_ids
        }

    @property
    def action_spaces(self):
        return {
            agent_id: self.action_space
            for agent_id in self.agent_ids
        }