import pygame
# ------------------------------------------------------------------------------
# Piece Selector UI: Anzeige der verfügbaren Steine und deren Transformationen
# ------------------------------------------------------------------------------
class Piece_selector_ui:
    def __init__(self, player, screen, cell_size=30):
        self.player = player
        self.screen = screen
        self.cell_size = cell_size
        self.font = pygame.font.Font(None, 36)
        # Der Selektor-Bereich, in dem die Steine angezeigt werden
        self.selector_area = pygame.Rect(650, 50, 200, 600)
        # Speichert für jeden gezeichneten Stein sein Rechteck zur Treffererkennung
        self.piece_rects = []

    def draw_selector(self):
        """Zeichnet den Selector-Bereich und alle verfügbaren Steine des Spielers."""
        pygame.draw.rect(self.screen, (220, 220, 220), self.selector_area)
        title = self.font.render("Deine Steine", True, (0, 0, 0))
        self.screen.blit(title, (self.selector_area.x + 10, self.selector_area.y + 10))
        # Vorherige Rechtecke löschen
        self.piece_rects = []
        padding = 10
        x_offset = self.selector_area.x + padding
        y_offset = self.selector_area.y + 50
        block_size = 15  # Größe eines Blocks, in dem ein Teil dargestellt wird

        for idx, piece in enumerate(self.player.piece_set):
            # Berechne die minimale und maximale Koordinate, um die Bounding Box zu bestimmen
            min_x = min(x for x, y in piece.shape)
            min_y = min(y for x, y in piece.shape)
            max_x = max(x for x, y in piece.shape)
            max_y = max(y for x, y in piece.shape)
            width = (max_x - min_x + 1) * block_size
            height = (max_y - min_y + 1) * block_size
            piece_rect = pygame.Rect(x_offset, y_offset, width, height)
            # Speichere das Rechteck zusammen mit dem zugehörigen Piece
            self.piece_rects.append((piece, piece_rect))
            # Zeichne einen Rahmen um die Bounding Box
            pygame.draw.rect(self.screen, (0, 0, 0), piece_rect, 2)
            # Zeichne die einzelnen Blöcke des Pieces
            for (x, y) in piece.shape:
                # Normiere, damit die Zeichnung immer oben links beginnt
                block_rect = pygame.Rect(x_offset + (x - min_x) * block_size,
                                         y_offset + (y - min_y) * block_size,
                                         block_size, block_size)
                pygame.draw.rect(self.screen, (100, 100, 100), block_rect)
            # Versetze y_offset, sodass der nächste Stein darunter gezeichnet wird
            y_offset += height + padding

    def get_selected_piece(self, mouse_pos):
        """
        Prüft, ob die Mausposition mouse_pos innerhalb eines
        der Rechtecke liegt, die die Steine repräsentieren.
        Gibt das zugehörige Piece zurück, oder None, wenn kein Treffer erfolgt.
        """
        for piece, rect in self.piece_rects:
            if rect.collidepoint(mouse_pos):
                return piece
        return None

    def draw_rotation_options(self, piece_transformations):
        """
        Zeigt alle möglichen Transformationen (z.B. Rotationen) für ein ausgewähltes Piece.
        piece_transformations: Liste von Piece-Objekten mit unterschiedlichen Transformationen.
        """
        # Zeichne die Transformationsoptionen in einem separaten Bereich (z. B. unterhalb des Selector-Bereichs)
        rotation_area = pygame.Rect(650, 400, 200, 200)
        pygame.draw.rect(self.screen, (200, 200, 200), rotation_area)
        title = self.font.render("Rotationen", True, (0, 0, 0))
        self.screen.blit(title, (rotation_area.x + 10, rotation_area.y + 10))
        padding = 10
        x_offset = rotation_area.x + padding
        y_offset = rotation_area.y + 50
        for trans in piece_transformations:
            # Zeichne jede Transformation in einer kleinen Vorschau
            for x, y in trans.shape:
                rect = pygame.Rect(x_offset + x * 15, y_offset + y * 15, 15, 15)
                pygame.draw.rect(self.screen, (150, 150, 150), rect)
            y_offset += 70