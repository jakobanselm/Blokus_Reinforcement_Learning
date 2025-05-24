from game.player import Player
from game.board import Board
from game.move_generator import Move_generator

# Beispielhafte Nutzung:
if __name__ == "__main__":
    # Vorausgesetzt, dass Board, Player und Piece bereits definiert sind.
    board = Board()
    player_y = Player("Y")
    move_gen = Move_generator(board)

    moves = move_gen.get_all_valid_moves(player_y)
    print("Gefundene gültige Züge:")
    for move in moves:
        original_piece, transformed_shape, origin = move
        print(f"Original: {original_piece.shape} -> Transformiert: {transformed_shape} bei Origin: {origin}")