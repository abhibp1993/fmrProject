"""
Author: Abhishek Kulkarni
Modified: November 14, 2016

Implements the greedy go-to-goal using high level specifications
given to car with partially observable environment.

Assumptions:
1. There exists a path to given goal location.
2. Obstacles are static.

The condition 2 is an extreme condition which reduces the problem to
simple A* like path finding algorithm. However, the intention of this
program is to establish the framework and test greedy strategy leads
to some solution.
"""

import numpy as np


class World(object):
    def __init__(self, dim=10):
        # Store local variables
        self.dim = dim          # Dimension of square world

        # Generate Label Map
        self.labelMap = self._generateLabels(dim)    # Generates labels to dim * dim world

        # Generate empty obstacle map
        self.obs = list()    #np.zeros((dim, dim), dtype=bool)

    def _generateLabels(self, dim):
        """
        Generates a map to label cells to integers.
        Example -- [(0, 2), (1, 2), (2, 2)              [6 7 8
                    (0, 1), (1, 1), (2, 1)     --->      3 4 5
                    (0, 0), (1, 0), (2, 0)]              0 1 2]

        @remark: numpy stores x, y axes differently as indicated below
                    |-----> Y
                    |
                   \/
                   X

        :param dim: Integer
        :return: 2D - numpy.array
        """
        # Initialize label map
        labelMap = np.zeros([dim, dim])
        labelMap[:] = np.NaN

        # Create labeling counter
        count = 0

        # Iteratively construct label map
        for i in range(dim):
            for j in range(dim):
                labelMap[j, i] = count
                count += 1

        return labelMap

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
        if row > self.dim or row < 0 or col > self.dim or col < 0:
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

    def obsByCell(self, cell):
        """
        Returns True if the given cell is occupied.

        :param cell: 2-tuple (x, y)
        :return: Boolean.
        :exception: IndexError, if index is out of bounds
        """
        return cell in self.obs      # self.obsMap[cell[0], cell[1]]

    def obsByLabel(self, label):
        """
        Returns True if the cell containing given label is occupied.

        :param label: Integer
        :return: Boolean.
        :exception: IndexError, if index is out of bounds
        """
        return self.obsByCell(self.cell(label))

    def addObstacleByCell(self, obsList):
        """
        Adds obstacles in the world.
        :param obsList: list of 2-tuples of format (x, y)
        :return: None
        @remark: No validation about the cell lying inside the world is done.
        """
        for c in obsList:
            self.obs.append(c)

    def addObstacleByLabel(self, obsList):
        """
        Adds obstacles in the world.
        :param obsList: list of integers.
        :return: None
        @remark: No validation about the label lying inside the world is done.
        """
        self.addObstacleByCell([self.cell(label) for label in obsList])

    def obsMap(self):
        """
        Returns the obstacle map as numpy array.
        :return: 2D numpy array of booleans.
        """
        # Initialize the matrix
        mat = np.zeros((self.dim, self.dim), dtype=np.bool)

        # Iteratively update the matrix
        for obs in self.obs:
            try:
                mat[obs[0], obs[1]] = True
            except IndexError:
                pass

        # Return the matrix
        return mat

    def slice(self, cell, dim):
        """
        Returns the label set of square slice of size=dim based at cell along with its occupancy matrix.

        :param cell: 2-tuple (x, y)
        :param dim: integer
        :return: 2-tuple (labelSet, obsSet)
        """
        # Initialize the labelSet and obsSet
        labelSet = np.zeros((dim, dim))
        obsSet = np.zeros((dim, dim), dtype=np.bool)
        labelSet[:] = np.NaN
        obsSet[:] = np.NaN

        # Check if the slice fits inside the map. If not trim the borders.
        row = cell[0]
        col = cell[1]
        maxRow = row + dim
        maxCol = col + dim

        if row < 0 or col < 0:
            raise IndexError('World.slice: x, y are out of bounds of world')

        if maxRow > self.dim:
            maxRow = self.dim

        if maxCol > self.dim:
            maxCol = self.dim

        # Slice off the labelSet
        labelSet = self.labelMap[row:maxRow, col:maxCol]

        # Modify the obsSet
        obsSet = self.obsMap()[row:maxRow, col:maxCol]

        # Return
        return labelSet, obsSet


if __name__ == '__main__':
    # Create World
    w = World(dim=3)
    w.addObstacleByCell([(1, 1), (0, 1)])
    w.addObstacleByLabel([7, 8])

    print(w.obsMap(), '\n', w.labelMap)

    # Test obsByCell and obsByLabel functions
    print('occupancy:', w.obsByCell(cell=(0, 0)), w.obsByLabel(label=8))

    # Slice World
    print(w.slice(cell=(0, 0), dim=2))