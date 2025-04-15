import pygame
from main_menu import Main_menu
from game_controller import Game_controller
# ------------------------------------------------------------------------------
# Hauptprogramm: Startet den UI-Flow (Main Menu -> Game Controller)
# ------------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("Blokus")

    # Main Menu anzeigen
    main_menu = Main_menu(screen)
    selected_mode = main_menu.run()
    if selected_mode is None:
        pygame.quit()
        return

    # Starte das Spiel (hier: 4 Spieler-Modus)
    game_controller = Game_controller(screen)
    game_controller.run()
    pygame.quit()


if __name__ == "__main__":
    main()