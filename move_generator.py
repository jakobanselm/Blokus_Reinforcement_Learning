from piece import Piece

# In move_generator.py
class Move_generator:
    def __init__(self, board):
        self.board = board

    def normalize_shape(self, shape):
        """
        Normalize a shape so that its minimum x and y coordinates start at (0,0).
        """
        min_x = min(x for x, y in shape)
        min_y = min(y for x, y in shape)
        return [(x - min_x, y - min_y) for x, y in shape]

    def generate_unique_transformations(self, piece):
        """
        Generate all unique rotations and reflections of a piece.
        Returns:
             a list of Piece objects with distinct normalized shapes.
             reflection_flag
             rotation amount
        """
        transformations = []
        seen = set()
        for reflect_flag in range(2):
            for rotations in range(4):
                # Copy the original shape
                temp_piece = Piece(list(piece.shape))
                piece_shape = temp_piece.get_positions((5,5),rotations, reflect_flag)
                # Normalize and sort shape for canonical representation
                norm_shape = self.normalize_shape(piece_shape)
                canon_shape = tuple(sorted(norm_shape))
                if canon_shape not in seen:
                    seen.add(canon_shape)
                    transformations.append([
                        Piece(list(canon_shape)),
                        rotations,
                        reflect_flag
                    ])
        return transformations

    def get_valid_origins(self, player, board = None):
        """
        Compute all board positions where the given player can start placing a piece.
        - For the first move: only empty corner cells.
        - For subsequent moves: empty cells that have diagonal contact but no edge contact.
        """

        valid_origins = set()
        if board is None:
            board = self.board
        else:
            board = board

        # First move: only empty corner positions
        if board.is_first_move(player):
            for pos in [
                (0, 0),
                (0, board.size - 1),
                (board.size - 1, 0),
                (board.size - 1, board.size - 1)
            ]:
                if board.is_empty(pos):
                    valid_origins.add(pos)
            return valid_origins

        # Subsequent moves
        for x in range(board.size):
            for y in range(board.size):
                if not board.is_empty((x, y)):
                    continue

                # Skip if any edge neighbor has the same color
                skip = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if board.in_bounds((nx, ny)) and board.grid[ny][nx] == player.color:
                        skip = True
                        break
                if skip:
                    continue

                # Include if any diagonal neighbor has the same color
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if board.in_bounds((nx, ny)) and board.grid[ny][nx] == player.color:
                        valid_origins.add((x, y))
                        break

        return valid_origins

    def get_moves_for_origin(self, player, origin):
        """
        Generate all valid moves for a given player and origin position.
        Returns a list of tuples:
          (X, Y, Piece index, roations, refelction_flag, candidate position list)
        """
        valid_moves = []
        # Iterate over available pieces and their indices via mask
        for piece_idx, piece in player.available_pieces():
            transforms = self.generate_unique_transformations(piece)
            for trans_piece in transforms:
                for pivot in trans_piece[0].shape:
                    # Compute translation vector to align pivot with origin
                    translation = (origin[0] - pivot[0], origin[1] - pivot[1])
                    # Compute absolute positions on board
                    candidate_positions = [
                        (b[0] + translation[0], b[1] + translation[1])
                        for b in trans_piece[0].shape
                    ]
                    # Check placement validity
                    if self.board.is_candidate_placement(candidate_positions, player):
                        valid_moves.append((
                            origin[0],
                            origin[1],
                            piece_idx,
                            trans_piece[1],
                            trans_piece[2],
                            candidate_positions))
        return valid_moves

    def get_valid_moves(self, player):
        """
        Generate all valid moves for a player by aggregating across all valid origins.
        Each move is represented as (origin_x, origin_y, piece_idx, rotation, reflect, absolute_coords_list).
        """
        valid_moves = []
        for origin in self.get_valid_origins(player):
            valid_moves.extend(self.get_moves_for_origin(player, origin))
        return valid_moves

