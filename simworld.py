#!/usr/bin/env python
"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 16, 2016

Defines grid-world for simulation.
"""

import numpy as np


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


# World
class World(object):
    """
    Defines a grid-world.
    Internally, the world is represented as a dictionary; <key, value> pairs,
        where key is the label of the cell (an integer) and the value is CellAP
        object containing boolean labels of atomic propositions over each cell.

    We use the following atomic propositions for now -
        {isRoad, isDynObs, isStatObs, stopSign}

    It is possible to increase the AP set in future by appending the new APs
    to label set.
    """
    def __init__(self, labels, cars=list()):
        """

        :param labels: is a dictionary of boolean numpy arrays (of size of world)
            representing the individual labels of each of the cell.
        :param cars: list of Car objects. (Default: empty list)
        """
        assert isinstance(labels, dict), 'World.__init__: ArgumentError: labels must be a dictionary'

        # Local Variable
        self.cars = cars

        # Validate label sets
        try:
            lastDim = None
            for k in labels.keys():

                # Get dimensions of labels set
                dim = labels[k].shape

                # Check for consistency of dimensions of labels
                if lastDim == None:
                    lastDim = dim
                else:
                    if lastDim != dim:
                        raise ValueError
                    else:
                        lastDim = dim

        except AttributeError:
            print('World.__init__: AttributeError in accessing dimension for label of key = ', k)

        except KeyError:
            print('World.__init__: KeyError in accessing labels. Check the dictionary', '\n', labels)

        except ValueError:
            print('World.__init__: ValueError in dimensions of labels. All labels must be of same size')

        # Generate World Representation
        self.labelMap = self._generateLabels(labels, lastDim)
        self.dim = lastDim

    def _generateLabels(self, labels, dim):
        """
        Generates world map using labels of cells.

        :param labels: is a dictionary of boolean numpy arrays (of size of world).
            It is assumed that keys are {'isRoad', 'stopSign', 'isStatObs'}

        :param dim: shape of numpy array of labels (rows, cols)-format
        :return: labeled dictionary <key=integer_labels, value=CellAP objects>

        @remark: This function is customized function based on how labels dictionary
            is defined during the execution. This function will change if the anything
            in labels dictionary definition changes.
        """
        # Copy dimensions
        ROWS, COLS = dim

        # Initialize the world map
        myMap = dict()

        # Iterate over cells and label set to construct world map.
        for r in range(ROWS):
            for c in range(COLS):
                # Instantiate new object
                myCell = CellAP()

                # Construct the CellAP object
                myCell.isRoad = bool(labels['isRoad'][r, c])
                myCell.isStatObs = bool(labels['isStatObs'][r, c])
                myCell.stopSign = bool(labels['stopSign'][r, c])

                # Update map
                myMap[c+r*COLS] = myCell

        return myMap

    def label(self, cell):
        """
        Returns the label of cell.

        :param cell: 2-tuple (x, y)
        :return: integer label of cell or np.NaN
        """
        assert isinstance(cell, (list, tuple)), 'World.label: cell must be a tuple/list'
        assert isinstance(cell, (list, tuple)), 'World.label: cell must be of size=2, format: (x, y)'

        # Extract x, y
        row = cell[0]
        col = cell[1]

        # Copy world size
        ROWS, COLS = self.dim

        # If cell is out of bounds, then return NaN
        if row >= ROWS or row < 0 or col >= COLS or col < 0:
            return np.NaN

        # Else: Return the label of cell
        return int(col+row*COLS)

    def cell(self, label):
        """
        Returns the cell of label.

        :param label: integer label of cell
        :return: 2-tuple (x, y)
        """
        assert isinstance(label, int), 'World.cell: label must be a integer.'

        # Copy world dimensions
        ROWS, COLS = self.dim

        # Check if label is outside world
        if label >= ROWS * COLS:
            return np.NaN, np.NaN

        # Compute the cell
        col = label % COLS
        row = int((label - col) / COLS)

        return row, col




# Define dummy map for testing
labels = {'isRoad': np.ones((4, 3)), 'isStatObs': np.ones((4, 3)), 'stopSign': np.zeros((4, 3))}

# Instantiate world
w = World(labels)

# Check map labels
for k in w.labelMap.keys():
    print(w.labelMap[k])

#  Check cell to label mapping
for r in range(4):
    for c in range(3):
        print((r, c), ' ', w.label((r, c)))

# Check label to cell mapping
for i in range(4*3):
    print(i, ' ', w.cell(i))