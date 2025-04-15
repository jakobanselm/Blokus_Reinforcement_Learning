class Board:
    def __init__(self, size=20):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def is_candidate_placement(self, positions, player):
        # Überprüft, ob alle Positionen innerhalb des Boards liegen und frei sind
        for pos in positions:
            if not self.in_bounds(pos) or not self.is_empty(pos):
                return False

        # Zusätzliche Regelprüfung:
        if self.is_first_move(player):
            # Beim ersten Zug muss mindestens eine der Positionen in einer der erlaubten Ecken liegen.
            allowed_starts = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
            if any(pos in allowed_starts for pos in positions):
                return True
            return False
        else:
            # Für Folgezüge: Muss diagonal an ein vorhandenes eigenes Piece anschließen, darf aber keine Kanten berühren.
            if not self.has_corner_contact(positions, player.color):
                return False
            if self.has_edge_contact(positions, player.color):
                return False
            return True

    def is_empty(self, pos):
        x, y = pos
        return self.grid[x][y] is None

    def is_first_move(self, player):
        """Überprüft, ob der Spieler noch keinen Zug gemacht hat."""
        for row in self.grid:
            for cell in row:
                if cell == player.color:
                    return False
        return True

    def has_corner_contact(self, positions, player_color):
        """Prüft, ob mindestens eine Ecke eines bereits platzierten Pieces berührt wird."""
        for x, y in positions:
            corner_positions = [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
            for cx, cy in corner_positions:
                if self.in_bounds((cx, cy)) and self.grid[cx][cy] == player_color:
                    return True
        return False

    def has_edge_contact(self, positions, player_color):
        """Prüft, ob eine Kante an ein bereits platziertes Piece angrenzend ist."""
        for x, y in positions:
            edge_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for ex, ey in edge_positions:
                if self.in_bounds((ex, ey)) and self.grid[ex][ey] == player_color:
                    return True
        return False

    def is_valid_placement(self, piece, positions, player):
        positions = positions
        print(f"positions to check: {positions}")
        # Zunächst: Alle Felder müssen innerhalb des Spielfelds liegen und frei sein.
        for pos in positions:
            if not self.in_bounds(pos) or not self.is_empty(pos):
                return False

        # Speziell für den ersten Zug: Das Piece muss den definierten Startpunkt abdecken.
        if self.is_first_move(player):
            # Hier wird als Startpunkt (0,0) angenommen. Passe das bei Bedarf an.
            if (self.size -1 ,self.size -1 ) in positions or (0 ,self.size -1 ) in positions or (self.size -1 ,0) in positions or (0,0) in positions:
                return True
            else:
                return False

        # Für nachfolgende Züge: Muss mindestens an einer Ecke Kontakt mit einem eigenen Piece haben...
        if not self.has_corner_contact(positions, player.color):
            return False

        # ... und darf nicht an einer Seite angrenzen.
        if self.has_edge_contact(positions, player.color):
            return False

        return True

    def place_piece(self, piece, positions, player):
        """Versucht, ein Piece zu platzieren. Bei Erfolg wird das Board aktualisiert."""
        if self.is_valid_placement(piece, positions, player):
            for x, y in positions:
                self.grid[x][y] = player.color
            return True
        return False

    def display(self):
        """Gibt das aktuelle Board in der Konsole aus."""
        for row in self.grid:
            print(" ".join([str(cell) if cell is not None else '.' for cell in row]))
        print()
