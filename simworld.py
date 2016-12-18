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

        # If cell is out of bounds, then return NaN
        if row >= self.dim or row < 0 or col >= self.dim or col < 0:
            return np.NaN

        # Else: Return the label of cell
        return int(self.labelMap[row, col])

    def cell(self, label):
        """
        Returns the cell of label.

        :param label: integer label of cell
        :return: 2-tuple (x, y)
        """
        assert isinstance(label, int), 'World.cell: label must be a integer.'

        # Else: Return the label of cell
        npCell = np.where(self.labelMap == label)
        cell = npCell[0].tolist() + npCell[1].tolist()
        if len(cell) == 0:
            cell = (np.NaN, np.NaN)

        return tuple(cell)

    def slice(self, cell, dim, direction):
        # Get label
        mylabel = self.label(cell)

        # transform the world
        if direction == NORTH:
            numRot = 0
        elif direction == SOUTH:
            numRot = 2
        elif direction == EAST:
            numRot = 1
        else:
            numRot = 3

        tmpWorld = copy.deepcopy(self.labelMap)
        tmpWorld = np.rot90(tmpWorld, numRot)

        npCell = np.where(tmpWorld == mylabel)
        cell = npCell[0].tolist() + npCell[1].tolist()
        if len(cell) == 0:
            cell = (np.NaN, np.NaN)

        cell_x = cell[0] - (dim - 1) / 2
        cell_y = cell[1]

        # Initialize the bounds
        slice_x_min = 0
        slice_x_max = dim
        slice_y_min = 0
        slice_y_max = dim
        world_x_min = cell_x
        world_x_max = cell_x + dim
        world_y_min = cell_y
        world_y_max = cell_y + dim

        # Check the X-bounds LOWER
        if cell_x < 0:
            slice_x_min = -cell_x
            world_x_min = 0

        # Check the Y-bounds LOWER
        if cell_y < 0:
            slice_y_min = -cell_y
            world_y_min = 0

        # Check the X-bounds UPPER
        if cell_x + dim > self.dim:
            slice_x_max = self.dim - cell_x
            world_x_max = self.dim

        # Check the Y-bounds UPPER
        if cell_y + dim > self.dim:
            slice_y_max = self.dim - cell_y
            world_y_max = self.dim

        # Initialize the slice
        sl = np.zeros((dim, dim))
        sl[:] = np.NaN

        # Update the slice
        sl[slice_x_min:slice_x_max, slice_y_min:slice_y_max] = \
            tmpWorld[world_x_min:world_x_max, world_y_min:world_y_max]

        # Return
        return sl

    def info(self, label):
        """
        Gets the data associated with the given cell.

        :param label: label of cell (Integer)
        :return: NodeAP instance.
        """
        x, y = self.cell(label)

        if np.NaN in [x, y]:
            print('Warning: Cell out of world!')
            return NodeAP()
        else:
            return NodeAP(yieldSign=self.yieldMap[x, y],
                          obstacle=self.obsMap[x, y],
                          stopSign=self.stopMap[x, y],
                          grass=self.grassMap[x, y],
                          road=self.roadMap[x, y])

    def dist(self, label1, label2):
        """
        Computes euclidean distance between 2 cells.

        :param label1: label of cell (integer)
        :param label2: label of cell (integer)
        :return: float or Inf
        """
        if label1 not in self or label2 not in self:
            return float('Inf')
        else:
            x1, y1 = self.cell(label1)
            x2, y2 = self.cell(label2)
            return np.linalg.norm([x1-x2, y1-y2])

    def _bmpParser(self, bmpImg):
        """
        Converts bitmap image into a numpy array.

        :param bmpImg: PIL.Image
        :return: 2D numpy.array, with usual --> Y  coordinates.
                                          X |
                                            \/
        @author: Ari, Yicong
        """
        # Read image
        im = Image.open(bmpImg, 'r')
        pix_val = list(im.getdata())

        # Extract information
        for pix in range(0, len(pix_val)):
            if pix_val[pix] > 0:
                pix_val[pix] = 0
            else:
                pix_val[pix] = 1

        # Return in appropriate format
        return np.reshape(np.array(pix_val), im.size)

    def _validateWorld(self):
        """
        Validates the dimension of all parsed bitmaps. Also checks if grass and road are not overlapping.

        :return: None.
        @author: Abhishek Kulkarni
        @:raises: AssertionError - If the conditions are not matched.
        """
        assert self.roadMap.size == self.grassMap.size and \
               self.grassMap.size == self.obsMap.size and \
               self.obsMap.size == self.yieldMap.size and \
               self.yieldMap.size == self.stopMap.size and \
               self.stopMap.size == self.dim**2, 'World._validateWorld: Dimensions of bitmaps and world dont match.'

        assert 1 not in np.bitwise_and(self.grassMap, self.roadMap), 'World._validateWorld: Grass and Roads overlap'

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.labelMap
        elif isinstance(item, (tuple, list)):
            return self.label(item) in self.labelMap
        else:
            return False


labels = {'isRoad': np.ones((4, 3)), 'isStatObs': np.ones((4, 3)), 'stopSign': np.zeros((4, 3))}
w = World(labels)
for k in w.labelMap.keys():
    print(w.labelMap[k])