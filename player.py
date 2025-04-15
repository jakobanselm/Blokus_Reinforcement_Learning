from pieces_defintion import PIECES_DEFINITION
from piece import Piece

class Player:
    def __init__(self, color):
        self.piece_set = []
        self.init_pieces()
        self.color = color

    def init_pieces(self):
        for piece_shape in PIECES_DEFINITION:
            self.piece_set.append(Piece(piece_shape))


    def drop_piece(self, piece):
        self.piece_set.remove(piece)