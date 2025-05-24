class Piece:
    def __init__(self, shape):
        """
        shape: List of tuples defining the relative positions of the squares,
               e.g. [(0, 0), (1, 0), (0, 1)] for an L-shaped piece.
        """
        self.shape = shape

    def __rotate_once(self, coords):
        return [(y, -x) for x, y in coords]

    def __reflect_once(self, coords):
        return [(-x, y) for x, y in coords]

    def get_positions(self, origin, rotation=0, reflect=0):
        """
        Return the absolute positions on the board when placing the piece at `origin`.
        rotation: number of 90° clockwise rotations (0–3)
        reflect: 0=no reflect, 1=reflect horizontally
        """
        # 1. Start from the original shape
        coords = list(self.shape)

        # 2. Apply rotations (mod 4)
        for _ in range(rotation % 4):
            coords = self.__rotate_once(coords)

        # 3. Apply a single reflection if requested
        if reflect:
            coords = self.__reflect_once(coords)

        # 4. Translate to board origin
        ox, oy = origin
        return [(ox + x, oy + y) for x, y in coords]

    def pretty_print(self, shape=None):
        """
        Print a graphical representation of the piece in the console.
        The piece’s squares are displayed in a grid.
        """
        if shape is None:
            shape = self.shape

        # Determine the minimum and maximum coordinates needed
        min_x = min(x for x, _ in shape)
        max_x = max(x for x, _ in shape)
        min_y = min(y for _, y in shape)
        max_y = max(y for _, y in shape)

        # Calculate grid dimensions: width and height
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Create an empty grid (list of lists) filled with spaces
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Fill the piece’s positions with "X" (or another marker)
        for x, y in shape:
            # Shift coordinates so the grid starts at (0, 0)
            grid[y - min_y][x - min_x] = "X"

        # Print the grid row by row
        for row in grid:
            print("".join(row))
