from blokus_env import BlokusEnv
from blokus_env_masked import Blokus_Env_Masked
from game import Game
from move_generator import Move_generator
from piece import Piece
from pieces_definition import PIECES_DEFINITION
import numpy as np
from numpy import random

game = Game(board_size = 14, player_colors=["X", "O"])


# 2) Environment aufbauen
env = Blokus_Env_Masked(game)

# 3) Reset und zufälligen Schritt zum Testen
obs = env.reset()
for _ in range(1000):
    act = random.choice(obs['valid_moves'])
    obs, r, done, info = env.step(act)
    env.render()
    print(r)
    print()
    if done:
        obs = env.reset()
print("Random rollout erfolgreich abgeschlossen.")

"""

player = game.players[0]
move_gen = Move_generator(board=game.board)
valid_moves = move_gen.get_valid_moves(player)
piece = Piece(PIECES_DEFINITION[19])
pos = piece.get_positions((1,3), 2,0)
game.board.place_piece(19,pos,player)
game.board.display()
print(f"place_piece: {game.board.place_piece(0,[(2,2)],player)}")
game.board.display()
valid_or = move_gen.get_valid_origins(player)
valid_moves = move_gen.get_valid_moves(player)

for move in valid_or:
    print(move)

print(len(valid_moves))
for move in valid_moves:
    print(move)



piece = Piece(PIECES_DEFINITION[19])
for i in range(0,4):
    for j in range(0,2):
        print(piece.get_positions((9, 9), i,j))
        piece.pretty_print()

"""






