from game import Game
from piece import Piece
from move_generator import Move_generator


def run_terminal_multistep_game_with_ui():
    # Spiel- und UI-Initialisierung
    game = Game(board_size=14, player_colors=["G", "Y"])
    move_gen = Move_generator(game.board)
    round_counter = 1

    while True:

        # 2) Terminal-Ausgabe
        current_player = game.players[game.current_player_index]
        print(f"\n---------- Runde {round_counter} ----------")
        print(f"Spieler {current_player.color} ist an der Reihe.")
        print("\nAktuelles Brett:")
        game.board.display()
        print("(Siehe UI-Fenster für aktuelle Brett-Ansicht.)")

        # 3) Schritt 1: Origins ermitteln
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
        origin_input = input("Bitte wähle den Index der Position (oder 'q'): ").strip()
        if origin_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            oi = int(origin_input)
            if oi < 0 or oi >= len(unique_origins):
                raise ValueError
        except ValueError:
            print("Ungültige Eingabe, bitte erneut.")
            continue
        selected_origin = unique_origins[oi]

        # 4) Schritt 2: Moves für diesen Origin
        valid_moves = move_gen.get_moves_for_origin(current_player, selected_origin)
        if not valid_moves:
            print("Keine Züge an dieser Position möglich, bitte erneut wählen.")
            continue

        # 5) Stück-Auswahl
        unique_pieces = []
        for m in valid_moves:
            piece, *_ = m
            if piece not in unique_pieces:
                unique_pieces.append(piece)
        print("\nVerfügbare Steine an dieser Position:")
        for idx, piece in enumerate(unique_pieces):
            print(f"{idx}: {piece.shape}")
            piece.pretty_print(None)
            print("-" * 20)
        piece_input = input("Index des Steins (oder 'q'): ").strip()
        if piece_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            pi = int(piece_input)
            if pi < 0 or pi >= len(unique_pieces):
                raise ValueError
        except ValueError:
            print("Ungültige Eingabe, bitte erneut.")
            continue
        selected_piece = unique_pieces[pi]

        # 6) Schritt 3: Transformation wählen
        moves_for_piece = [m for m in valid_moves if m[0] == selected_piece]
        unique_trans = []
        for m in moves_for_piece:
            _, shape, trans, _, cand = m
            if shape not in [u[0] for u in unique_trans]:
                unique_trans.append((shape, trans, cand))
        print("\nVerfügbare Transformationen:")
        for idx, (shape, trans, cand) in enumerate(unique_trans):
            print(f"{idx}: Positionen {cand} — Translation {trans}")
            selected_piece.pretty_print(cand)
            print("-" * 20)
        rot_input = input("Index der Transformation (oder 'q'): ").strip()
        if rot_input.lower() == 'q':
            print("Spiel beendet.")
            break
        try:
            ri = int(rot_input)
            if ri < 0 or ri >= len(unique_trans):
                raise ValueError
        except ValueError:
            print("Ungültige Eingabe, bitte erneut.")
            continue
        shape, trans, candidate_positions = unique_trans[ri]

        # 7) Zug ausführen und UI aktualisieren
        if game.board.place_piece(Piece(shape), candidate_positions, current_player):
            current_player.drop_piece(selected_piece)
            print("Zug erfolgreich!")

        else:
            print("Fehler: Zug ungültig.")

        game.next_turn()
        round_counter += 1

if __name__ == "__main__":
    run_terminal_multistep_game_with_ui()
