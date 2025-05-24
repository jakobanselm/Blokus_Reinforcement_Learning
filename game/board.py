class Board:
    def __init__(self, size=20):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def is_candidate_placement(self, positions, player):
        # Check that all positions are within the board and empty
        for pos in positions:
            if not self.in_bounds(pos) or not self.is_empty(pos):
                return False

        # Additional rule checks:
        if self.is_first_move(player):
            # On the first move, at least one of the positions must be in one of the allowed corners.
            allowed_starts = [
                (0, 0),
                (0, self.size - 1),
                (self.size - 1, 0),
                (self.size - 1, self.size - 1)
            ]
            return any(pos in allowed_starts for pos in positions)
        else:
            # For subsequent moves: must touch one of your own pieces by corner, but not by edge.
            if not self.has_corner_contact(positions, player.color):
                return False
            if self.has_edge_contact(positions, player.color):
                return False
            return True

    def is_empty(self, pos):
        x, y = pos
        return self.grid[y][x] is None

    def is_first_move(self, player):
        """Checks whether the player has not made any move yet."""
        for row in self.grid:
            for cell in row:
                if cell == player.color:
                    return False
        return True

    def has_corner_contact(self, positions, player_color):
        """Checks if at least one corner of the piece touches an existing piece of the same color."""
        for x, y in positions:
            corner_positions = [
                (x - 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x + 1, y + 1)
            ]
            for cx, cy in corner_positions:
                if self.in_bounds((cx, cy)) and self.grid[cy][cx] == player_color:
                    return True
        return False

    def has_edge_contact(self, positions, player_color):
        """Checks if any edge of the piece is adjacent to an existing piece of the same color."""
        for x, y in positions:
            edge_positions = [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1)
            ]
            for ex, ey in edge_positions:
                if self.in_bounds((ex, ey)) and self.grid[ey][ex] == player_color:
                    return True
        return False

    def is_valid_placement(self, piece_num, positions, player):

        # 1. Has the player still piece
        if player.pieces_mask[piece_num] == 0:
            return False

        # First: all cells must be within the board and empty
        for pos in positions:
            if not self.in_bounds(pos) or not self.is_empty(pos):
                return False

        # Special check for the first move: the piece must cover one of the defined start corners.
        if self.is_first_move(player):
            start_corners = [
                (0, 0),
                (0, self.size - 1),
                (self.size - 1, 0),
                (self.size - 1, self.size - 1)
            ]
            return any(corner in positions for corner in start_corners)

        # For subsequent moves: must have at least one corner contact...
        if not self.has_corner_contact(positions, player.color):
            return False

        # ... and must not have any edge contact.
        if self.has_edge_contact(positions, player.color):
            return False

        return True

    def place_piece(self, piece_num, positions, player):
        """Attempts to place a piece. If successful,remove piece from players pieces, updates the board."""
        #1. check if placement is valid
        if not self.is_valid_placement(piece_num, positions, player):
            return False
        else:
            player.pieces_mask[piece_num] = 0
            for x, y in positions:
                self.grid[y][x] = player.color
            return True

    def display(self):
        """Prints the current board to the console."""
        for row in self.grid:
            print(" ".join([str(cell) if cell is not None else '.' for cell in row]))
        print()
