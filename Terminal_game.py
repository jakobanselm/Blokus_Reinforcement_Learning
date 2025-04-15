# terminal_multistep_game.py
from game import Game
from piece import Piece
from move_generator import Move_generator

def run_terminal_multistep_game():
    game = Game(board_size=10, player_colors=["R", "B", "G", "Y"])
    move_gen = Move_generator(game.board)
    round_counter = 1

    while True:
        current_player = game.players[game.current_player_index]
        print(f"\n---------- Runde {round_counter} ----------")
        print(f"Spieler {current_player.color} ist an der Reihe.")
        print("\nAktuelles Brett:")
        game.board.display()

        # Schritt 1: Bestimme alle gültigen Origins (Kandidaten)
        valid_origins = move_gen.get_valid_origins(current_player)
        if not valid_origins:
            print(f"Kein gültiger Zug für Spieler {current_player.color} verfügbar!")
            game.next_turn()
            round_counter += 1
            continue

        unique_origins = sorted(valid_origins)
        print("\nMögliche Ursprungspositionen:")
        for idx, origin in enumerate(unique_origins):
            print(f"{idx}: {origin}")

        origin_input = input("Bitte wähle den Index der gewünschten Position (oder 'q' zum Beenden): ").strip()
        if origin_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            origin_index = int(origin_input)
            if origin_index < 0 or origin_index >= len(unique_origins):
                print("Ungültiger Index. Bitte versuche es erneut.")
                continue
        except ValueError:
            print("Bitte eine gültige Zahl eingeben!")
            continue
        selected_origin = unique_origins[origin_index]

        # Schritt 2: Bestimme alle gültigen Züge für den ausgewählten Ursprung
        valid_moves = move_gen.get_moves_for_origin(current_player, selected_origin)
        if not valid_moves:
            print("Für diesen Ursprung sind keine Züge möglich. Bitte wähle einen anderen Ursprung.")
            continue

        # Extrahiere eindeutige Pieces oder Transformationen (je nach gewünschter Detailtiefe)
        unique_pieces = []
        for move in valid_moves:
            piece, _, _, _, _ = move
            if piece not in unique_pieces:
                unique_pieces.append(piece)
        print("\nVerfügbare Steine an der gewählten Position:")
        for idx, piece in enumerate(unique_pieces):
            print(f"{idx}: {piece.shape}")
            piece.pretty_print(None)
            print("--------------------")
        piece_input = input("Bitte wähle den Index des gewünschten Steins (oder 'q' zum Beenden): ").strip()
        if piece_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            piece_index = int(piece_input)
            if piece_index < 0 or piece_index >= len(unique_pieces):
                print("Ungültiger Index. Bitte versuche es erneut.")
                continue
        except ValueError:
            print("Bitte eine gültige Zahl eingeben!")
            continue
        selected_piece = unique_pieces[piece_index]

        # Schritt 3: Für den ausgewählten Stein alle Transformationen ermitteln
        moves_for_piece = [move for move in valid_moves if move[0] == selected_piece]
        unique_transformations = []
        for move in moves_for_piece:
            piece, transformed_shape, translation, _, candidate_positions  = move
            if transformed_shape not in [t[0] for t in unique_transformations]:
                unique_transformations.append((piece, transformed_shape, translation, candidate_positions))
        if not unique_transformations:
            print("Keine Transformationen verfügbar. Bitte wähle einen anderen Stein.")
            continue

        print("\nVerfügbare Transformationen (Rotationen/Spiegelungen):")
        for idx, (piece, trans, translation, candidate_positions) in enumerate(unique_transformations):
            print(f"{idx}: {candidate_positions} mit Translation {translation}")
            piece.pretty_print(candidate_positions)
            print(f"-"*20)
        rotation_input = input("Bitte wähle den Index der gewünschten Transformation (oder 'q' zum Beenden): ").strip()
        if rotation_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            rotation_index = int(rotation_input)
            if rotation_index < 0 or rotation_index >= len(unique_transformations):
                print("Ungültiger Index. Bitte versuche es erneut.")
                continue
        except ValueError:
            print("Bitte eine gültige Zahl eingeben!")
            continue
        piece, selected_transformation, translation, candidate_positions = unique_transformations[rotation_index]
        # Führe den Zug aus:
        transformed_piece = Piece(selected_transformation)

        if game.board.place_piece(transformed_piece, candidate_positions, current_player):
            current_player.drop_piece(selected_piece)
            print("\nZug erfolgreich ausgeführt!")
        else:
            print("\nFehler: Der Zug konnte nicht ausgeführt werden.")

        game.next_turn()
        round_counter += 1

if __name__ == "__main__":
    run_terminal_multistep_game()