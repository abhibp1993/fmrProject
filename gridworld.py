"""
Author: Abhishek Kulkarni
Modified: November 14, 2016

Implements basic data structures world.
"""

import networkx as nx
import sm

GRASS = 0
ROAD = 1

ACT_EAST = lambda cell: (cell[0]+1, cell[1])
ACT_NORTH = lambda cell:(cell[0], cell[1]+1)
ACT_WEST = lambda cell: (cell[0]-1, cell[1])
ACT_NE = lambda cell: (cell[0]+1, cell[1]+1)
ACT_NW = lambda cell: (cell[0]-1, cell[1]+1)
ACTIONS = [ACT_NORTH, ACT_WEST, ACT_NE, ACT_EAST, ACT_NW]

NORTH = 0
EAST = 3
WEST = 1
SOUTH = 2

# Node: (location as 2-tuple, label)


class World(nx.DiGraph):
    def __init__(self, dim=10):
        super(World, self).__init__()

        self.dim = dim
        self.grid = self._createGrid()
        self._graphify()

    def __contains__(self, cell):
        return cell in self.nodes()

    def _createGrid(self):
        grid = list()
        for i in range(self.dim):
            for j in range(self.dim):
                grid.append(((i, j), GRASS))

        print len(grid), grid[1]
        return grid

    def _graphify(self):
        for cell in self.grid:
            self.add_node((cell, NORTH))
            self.add_node((cell, EAST))
            self.add_node((cell, WEST))
            self.add_node((cell, SOUTH))

        actions = [lambda cell: (cell[0]+1, cell[1]  ),         # east
                   lambda cell: (cell[0]+1, cell[1]+1),         # ne
                   lambda cell: (cell[0]  , cell[1]+1),         # north
                   lambda cell: (cell[0]-1, cell[1]+1),         # nw
                   lambda cell: (cell[0]-1, cell[1]  ),         # west
                   lambda cell: (cell[0]-1, cell[1]-1),         # sw
                   lambda cell: (cell[0]  , cell[1]-1),         # south
                   lambda cell: (cell[0]+1, cell[1]-1)]         # se

        for n in self.nodes():
            cell, dir = n
            if dir == NORTH:
                output = []


    def action(self, cell):
        return [ACT_EAST, ACT_NE, ACT_NORTH, ACT_NW, ACT_WEST]


class Car(sm.SM):
    def __init__(self, startState, world, goal):
        """

        :param startState: ((x, y), dir, speed)
        """
        self.startState = startState
        try:
            self.route = nx.shortest_path(world, source=startState[0], target=goal)
        except:
            self.route = list()

        print self.route

# run stuff
w = World(dim=5)
print w.number_of_nodes(), w.number_of_edges()
c = Car(((0, 0), NORTH, 1), w, (1, 1))


"""
test cases:
    1. c = Car(((1, 1), NORTH, 1), w, (0, 0)); print c.route --> should return empty list
    2. c = Car(((0, 0), NORTH, 1), w, (1, 1)); print c.route --> returns [(0, 0), (1, 1)]
"""