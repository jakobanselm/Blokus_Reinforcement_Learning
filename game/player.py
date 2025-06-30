from game.pieces_definition import PIECES_DEFINITION  # global list of all piece shapes
from game.piece import Piece
import numpy as np
import logging

class Player:
    def __init__(self, color):
        self.color = color
        # Binary mask: 1 = piece still available, 0 = piece already placed
        self.pieces_mask = np.ones(len(PIECES_DEFINITION), dtype=np.int8)

    def available_pieces(self):
        """
        Return a list of (index, Piece) tuples for every piece
        that is still available according to the mask.
        """
        return [
            (i, Piece(PIECES_DEFINITION[i]))
            for i in range(len(PIECES_DEFINITION))
            if self.pieces_mask[i] == 1
        ]

    def drop_piece(self, piece_idx):
        """
        Mark the piece at piece_idx as placed (i.e. no longer available).
        Raises IndexError if piece_idx is out of bounds.
        """
        if 0 <= piece_idx < len(self.pieces_mask):
            self.pieces_mask[piece_idx] = 0
        else:
            raise IndexError(f"No piece with index {piece_idx}")

    def reset_pieces(self):
        """
        Reset the availability mask so that all pieces become available again.
        Call this at the start of a new game/episode.
        """
        self.pieces_mask[:] = 1
