from enum import Enum

class Moves(Enum):
    WALK_LEFT=0
    WALK_RIGHT=1
    JUMP=2
    DROP=3
    SMASH_LEFT=4
    SMASH_RIGHT=5
    SMASH_UP=6
    SMASH_DOWN=7

    SPECIAL_LEFT=8
    SPECIAL_RIGHT=9
    SPECIAL_DOWN=10
    JAB=11

    WAIT = 12
    FOX_SPECIAL_DOWN=13
    FOX_RECOVERY=14


class CharacterMovesets:
    FOX= [Moves.JUMP, Moves.WALK_LEFT, Moves.WALK_LEFT, Moves.DROP, Moves.SMASH_DOWN, Moves.SMASH_LEFT, Moves.SMASH_RIGHT, Moves.SMASH_UP, Moves.FOX_SPECIAL_DOWN, Moves.SPECIAL_LEFT, Moves.SPECIAL_RIGHT, Moves.SMASH_UP, Moves.WAIT, Moves.FOX_RECOVERY]
