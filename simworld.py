#!/usr/bin/env python
"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 16, 2016

Defines grid-world for simulation.
"""


class CellAP:
    """
    Defines a structure representing state of a cell.
    Presently, we have 2 variables
        1. isRoad - whether cell is part of a road or is grass.
            (We may reasonably assume if cell is not road, then it's grass,
            i.e. it's undesirable to drive on it.)

        2. stopSign - whether there is a stop sign in a given cell.
        3. isDynObs - whether there is a dynamic obstacle in the cell.
        4. isStatObs - whether there is a static obstacle in the cell.
            ** Assuming separate knowledge about static and dynamic
            obstacles is reasonable because this information can be obtained
            from the vision system in real-world.

    In future, it might be possible to include other propositions about yield
    signs, traffic signals etc.
    """
    def __init__(self, isRoad=False, stopSign=False, isDynObs=False, isStatObs=False):
        """
        Initializes the structure.

        :param isRoad: Boolean. See class description.
        :param stopSign: Boolean. See class description.
        :param isDynObs: Boolean. See class description.
        :param isStatObs: Boolean. See class description.
        """
        self.isRoad = isRoad
        self.stopSign = stopSign
        self.isStatObs = isStatObs
        self.isDynObs = isDynObs

    def __str__(self):
        """
        Defines good printing string.
        """
        return '{road: ' + str(self.isRoad) + \
            ', stop: ' + str(self.stopSign) + \
            ', static obs: ' + str(self.isStatObs) + \
            ', dynamic obs: ' + str(self.isDynObs) + '}'


# Testing
a = CellAP()
b = CellAP(isDynObs=True)
c = CellAP(isRoad=True, stopSign=True, isStatObs=True, isDynObs=True)

print(a)
print(b)
print(c)
