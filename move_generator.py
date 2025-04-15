from piece import Piece
# In move_generator.py

class Move_generator:
    def __init__(self, board):
        self.board = board

    def normalize_shape(self, shape):
        min_x = min(x for x, y in shape)
        min_y = min(y for x, y in shape)
        return [(x - min_x, y - min_y) for x, y in shape]

    def generate_unique_transformations(self, piece):
        transformations = []
        seen = set()
        for reflect_flag in [False, True]:
            for rotations in range(4):
                new_shape = piece.shape[:]  # Kopie des Shapes
                temp_piece = Piece(new_shape)
                if reflect_flag:
                    temp_piece.reflect()
                for _ in range(rotations):
                    temp_piece.rotate()
                norm_shape = self.normalize_shape(temp_piece.shape)
                canon_shape = sorted(norm_shape)
                shape_tuple = tuple(canon_shape)
                if shape_tuple not in seen:
                    seen.add(shape_tuple)
                    transformations.append(Piece(canon_shape))
        return transformations

    def get_valid_origins(self, player):
        valid_origins = set()
        board = self.board
        if board.is_first_move(player):
            potential_origins = [
                (0, 0),
                (0, board.size - 1),
                (board.size - 1, 0),
                (board.size - 1, board.size - 1)
            ]
            for pos in potential_origins:
                if board.is_empty(pos):
                    valid_origins.add(pos)
        else:
            for x in range(board.size):
                for y in range(board.size):
                    pos = (x, y)
                    if board.is_empty(pos):
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            diag = (x + dx, y + dy)
                            if board.in_bounds(diag) and board.grid[diag[0]][diag[1]] == player.color:
                                valid_origins.add(pos)
                                break
        return valid_origins

    def get_moves_for_origin(self, player, origin):
        """
        Ermittelt für einen bestimmten Ursprung (origin) alle gültigen Züge.
        Dabei wird über alle Pieces und deren Transformationen (inklusive
        verschiedener Pivot-Punkte) iteriert.
        """
        valid_moves = []
        for piece in player.piece_set:
            transforms = self.generate_unique_transformations(piece)
            for trans_piece in transforms:
                for pivot in trans_piece.shape:
                    translation = (origin[0] - pivot[0], origin[1] - pivot[1])
                    candidate_positions = [(b[0] + translation[0], b[1] + translation[1])
                                           for b in trans_piece.shape]
                    if self.board.is_candidate_placement(candidate_positions, player):
                        valid_moves.append((piece, trans_piece.shape, translation, origin, candidate_positions))
        return valid_moves
