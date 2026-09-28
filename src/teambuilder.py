"""
Team pool for training. Loads every showdown paste in teams/ and hands them out in turn.
"""


import random
from pathlib import Path

from poke_env.teambuilder import Teambuilder


class VGCTeams(Teambuilder):
    """
    Loads all .txt teams from a folder, shuffles them with a seed and cycles through them.
    """

    def __init__(self, directory, seed=0):
        self.teams = [
            self.join_team(self.parse_showdown_team(p.read_text(encoding="utf-8")))
            for p in sorted(Path(directory).glob("*.txt"))
        ]
        random.Random(seed).shuffle(self.teams)
        self.index = 0

    def yield_team(self):
        team = self.teams[self.index % len(self.teams)]
        self.index += 1
        return team
