from game import Game
from board_ui import Board_ui
from move_generator import Move_generator
from piece_selector_ui import Piece_selector_ui
import pygame

# ------------------------------------------------------------------------------
# Game Controller: Steuert den Spielablauf und die mehrstufige Klickauswahl
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Game Controller: Steuert den Spielablauf und die mehrstufige Klickauswahl
# ------------------------------------------------------------------------------
class Game_controller:
    def __init__(self, screen):
        self.screen = screen
        self.game = Game()  # Initialisiere deine Spiel-Logik (4 Spieler, Board etc.)
        self.board_ui = Board_ui(self.game.board, screen)
        self.move_generator = Move_generator(self.game.board)
        # Wir gehen von 4 Spielern aus; platziere diese in den Ecken
        self.current_player = self.game.players[self.game.current_player_index]
        self.piece_selector_ui = Piece_selector_ui(self.current_player, screen)
        # Spielstufen: "position_selection", "piece_selection", "rotation_selection"
        self.stage = "position_selection"
        self.selected_position = None
        self.selected_piece = None
        self.selected_transformation = None

    def process_events(self, event):
        """Verarbeitet Mausklicks je nach aktuellem Auswahl-Schritt."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if self.stage == "position_selection":
                # Hier: Überprüfe, ob ein gültiger Platz angeklickt wurde
                valid_moves = self.move_generator.get_all_valid_moves(self.current_player)
                # (Beispielhaft: Überprüfe, ob mouse_pos in einem der hervorgehobenen Zellen liegt.)
                # Falls ja, setze self.selected_position und wechsle zur nächsten Stufe.
                self.selected_position = mouse_pos  # Platzhalter
                self.stage = "piece_selection"
            elif self.stage == "piece_selection":
                # Erfrage aus dem Selector, welcher Stein angeklickt wurde.
                piece = self.piece_selector_ui.get_selected_piece(mouse_pos)
                if piece is not None:
                    self.selected_piece = piece
                    # Erzeuge alle Transformationen für den gewählten Stein:
                    self.selected_transformation_options = self.move_generator.generate_unique_transformations(piece)
                    self.stage = "rotation_selection"
            elif self.stage == "rotation_selection":
                # Hier: Wähle anhand der angezeigten Transformationen die gewünschte aus.
                # Sobald der Spieler seine Auswahl trifft, führe den Zug aus:
                # (Zugausführung: board.place_piece(selected_transformation, selected_position, current_player))
                # und wechsle zur nächsten Runde.
                self.stage = "position_selection"
                # Aktualisiere den aktuellen Spieler, z. B.:
                self.game.next_turn()
                self.current_player = self.game.players[self.game.current_player_index]
                self.piece_selector_ui = Piece_selector_ui(self.current_player, self.screen)

    def update(self):
        """Aktualisiert die Anzeige basierend auf der aktuellen Stufe."""
        self.board_ui.draw_board()
        if self.stage == "position_selection":
            valid_moves = self.move_generator.get_all_valid_moves(self.current_player)
            self.board_ui.highlight_valid_positions(valid_moves)
        elif self.stage == "piece_selection":
            self.piece_selector_ui.draw_selector()
        elif self.stage == "rotation_selection":
            self.piece_selector_ui.draw_rotation_options(self.selected_transformation_options)
        pygame.display.flip()

    def run(self):
        """Hauptspielschleife für die Pygame UI."""
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.process_events(event)
            self.update()
            clock.tick(30)