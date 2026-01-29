"""Constants for PettingZoo Arena integration."""

from typing import Dict, List

# Action meanings derived from ALE inspection
STANDARD_18 = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]
PONG_6 = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
OTHELLO_10 = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]
SURROUND_5 = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]
VIDEO_CHECKERS_5 = ["FIRE", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]
WIZARD_10 = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE"]

ACTION_MEANINGS: Dict[str, List[str]] = {
    "basketball_pong": PONG_6,
    "boxing": STANDARD_18,
    "combat_plane": STANDARD_18,
    "combat_tank": STANDARD_18,
    "double_dunk": STANDARD_18,
    "entombed_competitive": STANDARD_18,
    "entombed_cooperative": STANDARD_18,
    "flag_capture": STANDARD_18,
    "foozpong": STANDARD_18,
    "ice_hockey": STANDARD_18,
    "joust": STANDARD_18,
    "mario_bros": STANDARD_18,
    "maze_craze": STANDARD_18,
    "othello": OTHELLO_10,
    "pong": PONG_6,
    "space_invaders": PONG_6,
    "space_war": STANDARD_18,
    "surround": SURROUND_5,
    "tennis": STANDARD_18,
    "video_checkers": VIDEO_CHECKERS_5,
    "volleyball_pong": PONG_6,
    "wizard_of_wor": WIZARD_10
}
