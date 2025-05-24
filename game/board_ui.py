import pygame

# ------------------------------------------------------------------------------
# Board UI: Darstellung des Spielbretts und Hervorhebung gültiger Züge
# ------------------------------------------------------------------------------
class Board_ui:
    def __init__(self, board, screen, cell_size=30):
        self.board = board
        self.screen = screen
        self.cell_size = cell_size
        self.board_color = (240, 240, 240)
        self.grid_color = (50, 50, 50)
        # Farbkodierung für Spieler: z.B. "R", "B", "G", "Y"
        self.player_colors = {
            "R": (255, 0, 0),
            "B": (0, 0, 255),
            "G": (0, 255, 0),
            "Y": (255, 255, 0)
        }

    def draw_board(self):
        """Zeichnet das Spielfeld inklusive Gitterlinien."""
        self.screen.fill(self.board_color)
        board_size = self.board.size
        for row in range(board_size):
            for col in range(board_size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)
                # Falls in der Zelle ein Spieler steht, fülle die Zelle in dessen Farbe:
                cell_value = self.board.grid[row][col]
                if cell_value is not None:
                    color = self.player_colors.get(cell_value, (0, 0, 0))
                    pygame.draw.rect(self.screen, color, rect)

    def highlight_valid_positions(self, valid_moves):
        """
        Hebt alle Positionen hervor, an denen ein Zug möglich ist.
        valid_moves: Liste von (piece, transformed_shape, origin) Tupeln.
        """
        # Wir extrahieren die Origin-Punkte (x, y) und heben diese Zellen hervor.
        highlighted = set()
        for _, _, origin in valid_moves:
            highlighted.add(origin)

        highlight_color = (0, 255, 255)  # Cyan als Beispiel
        for origin in highlighted:
            x, y = origin
            rect = pygame.Rect(y * self.cell_size, x * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, highlight_color, rect, 3)