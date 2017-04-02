from Ball import Ball
from Player import Player


def _convert_type(inp):
	if inp==None:
		ret = 0.0
	else:
		ret = float(inp)
	return ret
class Moment:
    """A class for keeping info about the moments"""
    def __init__(self, moment):
        self.quarter = moment[0]  # Hardcoded position for quarter in json
        self.game_clock = moment[2]  # Hardcoded position for game_clock in json
        self.shot_clock = _convert_type(moment[3])  # Hardcoded position for shot_clock in json
        ball = moment[5][0]  # Hardcoded position for ball in json
        self.ball = Ball(ball)
        players = moment[5][1:]  # Hardcoded position for players in json
        self.players = [Player(player) for player in players]