class Piece:
    def __init__(self, shape):
        """
        shape: Liste von Tupeln, die die relativen Positionen der Felder definieren,
               z.B. [(0,0), (1,0), (0,1)] für ein L-förmiges Teil.
        """
        self.shape = shape

    def rotate(self):
        """Dreht das Teil um 90 Grad im Uhrzeigersinn."""
        self.shape = [(y, -x) for x, y in self.shape]

    def reflect(self):
        """Spiegelt das Teil horizontal."""
        self.shape = [(-x, y) for x, y in self.shape]

    def get_positions(self, origin):
        """
        Gibt die absoluten Positionen auf dem Brett zurück, wenn das Teil an 'origin' platziert wird.
        """
        return [(origin[0] + x, origin[1] + y) for x, y in self.shape]

    def pretty_print(self, shape):
        """
        Gibt eine grafische Darstellung des Pieces in der Konsole aus.
        Dabei werden die Felder des Pieces in einem Raster angezeigt.
        """
        if not shape:
            shape = self.shape


        # Bestimme minimal und maximal benötigte Koordinaten
        min_x = min(x for x, _ in shape)
        max_x = max(x for x, _ in shape)
        min_y = min(y for _, y in shape)
        max_y = max(y for _, y in shape)

        # Größe des Rasters: Breite und Höhe
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Erzeuge ein leeres Raster (Liste von Listen) mit Leerzeichen
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Fülle die Positionen des Pieces mit "X" (oder einem anderen Zeichen)
        for (x, y) in shape:
            # Verschiebe Koordinaten so, dass sie bei 0 beginnen
            grid[y - min_y][x - min_x] = "X"

        # Gib das Raster zeilenweise aus
        for row in grid:
            print("".join(row))
