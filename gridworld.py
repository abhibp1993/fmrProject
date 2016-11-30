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

NORTH = 1
EAST = 0
WEST = 2
SOUTH = 3


class World(nx.DiGraph):
    def __init__(self, dim=10):
        super(World, self).__init__()

        self.dim = dim
        self.grid = self._createGrid()
        self._graphify()

    def __contains__(self, cell):
        cells = [i[0] for i in self.nodes()]
        return cell in cells

    def _createGrid(self):
        grid = list()
        for i in range(self.dim):
            for j in range(self.dim):
                grid.append(((i, j), GRASS))

        print len(grid), grid[1]
        return grid

    def isGrass(self, cell):
        for i in self.nodes():
            if i[0] == cell:
                return i[1]

    def _graphify(self):
        # Generate all nodes
        for cell in self.grid:
            self.add_node(cell + (NORTH,))
            self.add_node(cell + (EAST,))
            self.add_node(cell + (WEST,))
            self.add_node(cell + (SOUTH,))

        # List all possible action
        actions = list()
        actions.append(lambda cell: (cell[0]+1, cell[1]))           # east
        actions.append(lambda cell: (cell[0]+1, cell[1]+1))         # ne
        actions.append(lambda cell: (cell[0]  , cell[1]+1))         # north
        actions.append(lambda cell: (cell[0]-1, cell[1]+1))         # nw
        actions.append(lambda cell: (cell[0]-1, cell[1]))           # west
        actions.append(lambda cell: (cell[0]-1, cell[1]-1))         # sw
        actions.append(lambda cell: (cell[0]  , cell[1]-1))         # south
        actions.append(lambda cell: (cell[0]+1, cell[1]-1))         # se

        # Perform each action on node, and add new edges
        for n in self.nodes():
            for act in actions:
                newCell = act(n[0])
                if newCell not in self:
                    continue

                newNode = (newCell, n[1], n[2])

                idx = actions.index(act)
                if   n[2] == NORTH and idx == 0: newNode = (newCell, self.isGrass(newCell), EAST)
                elif n[2] == NORTH and idx == 4: newNode = (newCell, self.isGrass(newCell), WEST)

                if   n[2] == WEST and idx == 2: newNode = (newCell, self.isGrass(newCell), NORTH)
                elif n[2] == WEST and idx == 6: newNode = (newCell, self.isGrass(newCell), SOUTH)

                if   n[2] == SOUTH and idx == 4: newNode = (newCell, self.isGrass(newCell), WEST)
                elif n[2] == SOUTH and idx == 0: newNode = (newCell, self.isGrass(newCell), EAST)

                if   n[2] == EAST and idx == 6: newNode = (newCell, self.isGrass(newCell), SOUTH)
                elif n[2] == EAST and idx == 2: newNode = (newCell, self.isGrass(newCell), NORTH)

                try:
                    self.add_edge(n, newNode, weight=idx)
                except:
                    pass

    def action(self, cell):
        return [ACT_EAST, ACT_NE, ACT_NORTH, ACT_NW, ACT_WEST]


class Car(sm.SM):
    def __init__(self, startState, world, goal):
        """

        :param startState: ((x, y), dir, speed)
        :param world: World object
        :param goal: 2-tuple
        """
        self.startState = startState
        try:
            self.route = nx.shortest_path(world, source=startState, target=(goal, GRASS, NORTH))
        except Exception, e:
            print e
            self.route = list()

        print self.route

    def getNextValues(self, state, inp):
        """

        :param state:
        :param inp:
        :return:
        """
        # Get next action
        nextDesiredAction = self.route.pop(0)

        #



# run stuff
w = World(dim=5)
print w.number_of_nodes(), w.number_of_edges()
#print w.nodes()
c = Car(startState=((0, 0), GRASS, NORTH), world=w, goal=(1, 1))


"""
test cases:
    1. c = Car(((1, 1), NORTH, 1), w, (0, 0)); print c.route --> should return empty list
    2. c = Car(((0, 0), NORTH, 1), w, (1, 1)); print c.route --> returns [(0, 0), (1, 1)]
"""