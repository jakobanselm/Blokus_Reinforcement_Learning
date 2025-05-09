from player import Player
from board import Board

class Game:
    def __init__(self, board_size=20, player_colors=["R", "B", "G", "Y"]):
        self.board = Board(board_size)
        # Create player objects based on the defined colors.
        self.players = [Player(color) for color in player_colors]
        self.current_player_index = 0

    def play_turn(self):
        current_player = self.players[self.current_player_index]
        print(f"Player {current_player.color}’s turn.")

        # Normally here would be input (e.g., via console or GUI)
        # For this example, simply take the first available piece.
        if not current_player.pieces_mask:
            print(f"Player {current_player.color} has no pieces left!")
            self.next_turn()
            return

        piece = current_player.pieces_mask[0]
        origin = (5, 5)  # Sample placement; in a real game determined by input

        if self.board.place_piece(piece, origin, current_player):
            print(f"Player {current_player.color} successfully places a piece.")
            current_player.drop_piece(piece)
        else:
            print(f"Invalid move for player {current_player.color}.")
            # Here you could trigger another input attempt.

        self.board.display()
        self.next_turn()

    def next_turn(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
