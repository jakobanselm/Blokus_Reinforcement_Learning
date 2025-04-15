import pygame

# ------------------------------------------------------------------------------
# Main Menu: Startfenster zur Auswahl des Spielmodus
# ------------------------------------------------------------------------------
class Main_menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 48)
        self.selected_mode = None

    def run(self):
        """Zeigt das Menü, bis der Benutzer den Spielmodus auswählt."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.selected_mode = None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Für dieses Beispiel: Jeder Klick wählt den 4-Spieler-Modus aus.
                    self.selected_mode = "4_player"
                    running = False

            self.screen.fill((200, 200, 200))
            title_text = self.font.render("Blokus - Spielmodus wählen", True, (0, 0, 0))
            mode_text = self.font.render("4 Spieler", True, (0, 0, 0))
            self.screen.blit(title_text, (50, 50))
            self.screen.blit(mode_text, (50, 150))
            pygame.display.flip()

        return self.selected_mode