from player import Player
from board import Board

class Game:
    def __init__(self, board_size=20, player_colors=["R", "B", "G", "Y"]):
        self.board = Board(board_size)
        # Erzeuge Spieler-Objekte anhand der definierten Farben.
        self.players = [Player(color) for color in player_colors]
        self.current_player_index = 0

    def play_turn(self):
        current_player = self.players[self.current_player_index]
        print(f"Spieler {current_player.color} ist an der Reihe.")

        # Hier wäre normalerweise die Eingabe (z. B. per Konsole oder GUI)
        # Für dieses Beispiel wird einfach das erste verfügbare Piece genommen.
        if not current_player.piece_set:
            print(f"Spieler {current_player.color} hat keine Pieces mehr!")
            self.next_turn()
            return

        piece = current_player.piece_set[0]
        origin = (5, 5)  # Beispielhafte Platzierung; in einem echten Spiel per Eingabe bestimmt

        if self.board.place_piece(piece, origin, current_player):
            print(f"Spieler {current_player.color} platziert erfolgreich ein Piece.")
            current_player.drop_piece(piece)
        else:
            print(f"Ungültiger Zug für Spieler {current_player.color}.")
            # Hier könnte man eine erneute Eingabe anstoßen.

        self.board.display()
        self.next_turn()

    def next_turn(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

