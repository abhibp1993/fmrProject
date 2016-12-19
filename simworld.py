#!/usr/bin/env python
"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 16, 2016

Defines grid-world for simulation.
"""

import numpy as np
from copy import deepcopy


# Class CellAP
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


# Class World
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
        self.world = self._generateLabels(labels, lastDim)
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

    # TODO: Test getDynObsMap function after definition, integration of Car class.
    def getDynObsMap(self):
        """
        Returns boolean numpy array of dynamic obstacles.
        :return: bool numpy.array
        """
        # Initialize map
        obsMap = np.zeros(shape=self.dim, dtype=np.bool)

        # Modify map to include obstacles
        for car in self.cars:
            carPosition = car.position      # returns label of cell occupied
            obsMap[self.cell(carPosition)] = 1

        return obsMap

    def getStatObsMap(self):
        """
        Returns boolean numpy array of static obstacles.
        :return: bool numpy.array
        """
        # Initialize map
        statObsMap = np.zeros(shape=self.dim, dtype=np.bool)

        # Update map
        for label in self.world.keys():
            statObsMap[self.cell(label)] = self.world[label].isStatObs

        return statObsMap

    def getStopSignMap(self):
        """
        Returns boolean numpy array of stop signs.
        :return: bool numpy.array
        """
        # Initialize map
        stopMap = np.zeros(shape=self.dim, dtype=np.bool)

        # Update map
        for label in self.world.keys():
            stopMap[self.cell(label)] = self.world[label].stopSign

        return stopMap

    def getRoadMap(self):
        """
        Returns boolean numpy array of roads.
        :return: bool numpy.array
        """
        # Initialize map
        roadMap = np.zeros(shape=self.dim, dtype=np.bool)

        # Update map
        for label in self.world.keys():
            roadMap[self.cell(label)] = self.world[label].isRoad

        return roadMap

    def labelMap(self):
        """
        Returns labeled world.
        :return: bool numpy.array
        """
        # Initialize map
        myWorld = np.zeros(shape=self.dim, dtype=np.uint8)

        # Label the world
        ROWS, COLS = self.dim
        for r in range(ROWS):
            for c in range(COLS):
                myWorld[r, c] = self.label((r, c))

        return myWorld

    def slice(self, cell, angle, size):
        """
        Returns the slice of world with
            1. left-bottom corner of slice placed on cell specified.
            2. x-axis of slice at angle specified as parameter.
            3. dimension of slice specified by size parameter.

        Convention: X-axis (0 deg) is aligned along columns.

        :param cell: integer label of cell
        :param angle: multiple of 90 in range [0, 360)
        :param size: 2-tuple size (row, col) of slice.
        :return:

        @remark: extend angle of slice to any value in [0, 360) angle in radians
        """
        assert angle % 90 == 0 and 360 > angle >= 0, 'World.slice: angle must be in [0, 360) range and multiple of 90.'

        # Get the base cell.
        cellR, cellC = cell
        width, height = size

        # Validate base point
        SLICE_ROWS, SLICE_COLS = size
        WORLD_ROWS, WORLD_COLS = self.dim
        if cellR >= WORLD_ROWS or cellC >= WORLD_COLS or (cellR + width) <= 0 or (cellC + height) <= 0:
            raise AssertionError('World.slice: Slice does not fit inside world.')

        # Rotate World in reverse way instead of rotating the window
        rotAngle = int((360 - angle) / 90)
        world = self.labelMap()
        world = np.rot90(world, rotAngle)

        # Define slicing bounds
        slice_row_min = max(0, -cellR)
        slice_col_min = max(0, -cellC)
        slice_row_max = min(WORLD_ROWS - cellR, SLICE_ROWS)
        slice_col_max = min(WORLD_COLS - cellC, SLICE_COLS)

        world_row_min = max(0, cellR)
        world_col_min = max(0, cellC)
        world_row_max = min(WORLD_ROWS, SLICE_ROWS + cellR)
        world_col_max = min(WORLD_COLS, SLICE_COLS + cellC)

        # Initialize the slice
        mySlice = np.zeros(size)
        mySlice[:] = np.NaN

        # Slice the world!
        mySlice[slice_row_min:slice_row_max, slice_col_min:slice_col_max] = \
           world[world_row_min:world_row_max, world_col_min:world_col_max]

        return mySlice



# Function: parseMaps
def parseMaps(bmpRoad=None, bmpStopSign=None, bmpStatObs=None):
    """
    Parses the bitmaps and generates binary numpy 2D-arrays to generate world.

    :param bmpRoad: binary bitmap image
    :param bmpStopSign: binary bitmap image
    :param bmpStatObs: binary bitmap image
    :return: dictionary of form {'isRoad': np.array(x, y), 'isStatObs': np.array(x, y), 'stopSign': np.array(x, y)}
    """
    pass


# Main code
if __name__ == '__main__':
    # Define dummy map for testing
    labels = {'isRoad': np.ones((4, 3)), 'isStatObs': np.ones((4, 3)), 'stopSign': np.zeros((4, 3))}

    # Instantiate world
    w = World(labels)

    # Check map labels
    for k in w.world.keys():
        print(w.world[k])

    #  Check cell to label mapping
    for r in range(4):
        for c in range(3):
            print((r, c), ' ', w.label((r, c)))

    # Check label to cell mapping
    for i in range(4*3):
        print(i, ' ', w.cell(i))

    # Check getStatObsMap
    print('getStatObs-----------')
    print(w.getStatObsMap())

    # Check getStopSignMap
    print('getStopSign-----------')
    print(w.getStopSignMap())

    # Check getRoadMap
    print('getRoadMap-----------')
    print(w.getRoadMap())

    # Check Slicing
    print('Slicing-----------')
    print(w.labelMap())
    print(w.slice((-1, -1), 0, (3, 3)))
