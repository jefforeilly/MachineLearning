#!/usr/bin/env python
# -*- coding: utf-8  -*-
"""Numpy implementation of multi-player gamblers ruin game.

Possible modifications:
    - different probability of winning for each player
    - maximum number of turns
    - saves _history in GamblersRuin object (2D np.array)
    - different capital for each player

"""

import sys

import numpy as np

DEFAULT_CAPITAL = 40


def _print_err(message: str):
    """Print message to stderr.

    Parameters
    ----------
    message: str
           Message printed to error stream

    Returns
    -------
    exit(1)
         Returns with code 1 for error

    """
    print(message, file=sys.stderr)
    exit(1)


class GamblersRuin:
    """GamblersRuin class allowing playing and modifying settings."""

    def __init__(self,
                 n_players: int,
                 *,
                 probabilities=None,
                 capitals=None,
                 strategy: str = 'all'):
        """Parse possible modifications to the game.

        Possible modifications:
            - different probability of winning for each player
            - maximum number of turns
            - saves _history in GamblersRuin object (2D np.array)
            - different capital for each player

        Bankruptcy boundary is set to 0 points

        Parameters
        ----------
        n_players: : int
               Number of player participating in gambler's game
        probabilities: list-like, optional
               List-like object containing probability of winning for each
               player.

               DEFAULT: Equal probability for each player

               IMPORTANT: Array doesn't have to sum to one; each weight is
               divided by the sum of all elements in each turn
        capitals: list-like, optional
               List-like object containing integers, starting capital for each
               player.

               DEFAULT: Same as DEFAULT_CAPITAL global variable
               (default: 40 for each player)

        strategy: str, optional
               String specifying strategy of gaining points after winning.
               Two possibilities only:
                  - 'all' - each player gives one point to the winner
                  - 'one' - winning player gets one point

               IMPORTANT: Each player loses one point for losing a round

        """
        # If probability not specified use uniformly distributed weighted array
        if probabilities is None:
            self.probabilities = np.full(n_players, 1 / n_players)
        else:
            if len(probabilities) != n_players:
                _print_err('Number of players different than length of'
                           'probabilities list')
            else:
                self.probabilities = probabilities

        # If capital is None give each player the same amount of points
        # by default DEFAULT_CAPITAL (equal 40)
        if capitals is None:
            self.capitals = np.full(n_players, DEFAULT_CAPITAL)
        else:
            if len(capitals) != n_players:
                _print_err('Number of players different than length of'
                           'starting capitals list')
            else:
                self.capitals = capitals

        # If strategy different than one or all [see docs] raise error
        if strategy != 'one' and strategy != 'all':
            _print_err(
                'Unknown strategy, use "one" (player gains 1 for win) or'
                'or "all" (player gains n-1 for win)')
        else:
            self.strategy = strategy

        self._history = []

    def play(self, max_rounds: int = 10000):
        """Play one round of Gambler's game.

        Game ends if:
        - everyone except one player gone bankrupt
        - max_rounds turns were played and there was no winner

        Parameters
        ----------
        max_rounds : int
             Maximum number of rounds allowed for this play

        Returns
        -------
        np.array
             Array of shape (turns, players) containg history of each round
             (e.g. first row is the first round, second the second etc.)

        """
        # Play at most max_rounds
        for _ in range(max_rounds):
            # If only one player left, end the play
            if np.count_nonzero(self.capitals) == 1:
                break

            # Get random winner based on weighted probability array
            winner = np.random.choice(
                len(self.probabilities),
                p=self.probabilities / np.sum(self.probabilities))

            # add capital for the winner (add 1 or n_players-1)
            if self.strategy == 'one':
                self.capitals[winner] += 1
            else:
                self.capitals[winner] += np.count_nonzero(self.capitals) - 1

            # Drop capital of everyone except the winner
            # but only if player is in the game (if his capital != 0)

            self.capitals[np.logical_and(
                np.arange(len(self.probabilities)) != winner,
                np.where(self.capitals != 0, True, False))] -= 1

            # Remove players with 0 capital from the game
            # (assign 0 probability of their win)
            self.probabilities = np.where(self.capitals == 0, 0,
                                          self.probabilities)
            # Save current round in _history
            self._history.append(np.copy(self.capitals))

        return np.array(self._history)

    @property
    def history(self):
        """Return history of last play.

        Returns
        -------
        np.array
             Array of shape (turns, players) containg history of each round
             (e.g. first row is the first round, second the second etc.)

        """
        return np.array(self._history)


if __name__ == '__main__':
    GR = GamblersRuin(5)
    GR.play(100000)
    print(GR.history[-1])
