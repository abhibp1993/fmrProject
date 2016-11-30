"""
Author: Ari Goodman, Abhishek Kulkarni
Modified: November 30, 2016

Creates labeled grid with labels for nodes and edges and uses A* to solve.
"""

import networkx as nx
import sm
import labeledEdge

EAST = 0
NORTH = 1
WEST = 2
SOUTH = 3

ACT_EAST = lambda cell: (cell[0]+1, cell[1])
ACT_NORTH = lambda cell: (cell[0], cell[1]+1)
ACT_WEST = lambda cell: (cell[0]-1, cell[1])
ACT_NE = lambda cell: (cell[0]+1, cell[1]+1)
ACT_NW = lambda cell: (cell[0]-1, cell[1]+1)
ACT_WAIT = lambda cell: (cell[0]-1, cell[1])
ACTIONS = [ACT_NORTH, ACT_WEST, ACT_NE, ACT_EAST, ACT_NW, ACT_WAIT]

class World(nx.DiGraph):
    def __init__(self, dim=10):
        super(World, self).__init__()
        self.dim = dim
        self.grid = self._createGrid()
        self._graphify()

    def __contains__(self, cell):
        cells = [i[0:2] for i in self.nodes()]
        return cell in cells

    def _createGrid(self):
        grid = list()
        for i in range(self.dim):
            for j in range(self.dim):
                grid.append((i, j))
        return grid

    def _graphify(self):
        # Generate all nodes
        for cell in self.grid:
            self.add_node(cell + (NORTH,))
            self.add_node(cell + (EAST,))
            self.add_node(cell + (WEST,))
            self.add_node(cell + (SOUTH,))

        print self.nodes()
        # List all possible action
        actions = list()
        #print (cell[0]+1, cell[1])

        actions.append(lambda cell: (cell[0]+1, cell[1]))           # east
        actions.append(lambda cell: (cell[0]+1, cell[1]+1))         # ne
        actions.append(lambda cell: (cell[0]  , cell[1]+1))         # north
        actions.append(lambda cell: (cell[0]-1, cell[1]+1))         # nw
        actions.append(lambda cell: (cell[0]-1, cell[1]))           # west
        actions.append(lambda cell: (cell[0]-1, cell[1]-1))         # sw
        actions.append(lambda cell: (cell[0]  , cell[1]-1))         # south
        actions.append(lambda cell: (cell[0]+1, cell[1]-1))         # se
        actions.append(lambda cell: (cell[0]  , cell[1]))           # wait

        # Perform each action on node, and add new edges
        for n in self.nodes():
            for act in actions:
                newCell = act(n[0:2])
                if newCell not in self:
                    continue

                newNode = (newCell) + (n[1],)

                idx = actions.index(act)
                if   n[1] == NORTH and idx == 0: newNode = (newCell) + (EAST,)
                elif n[1] == NORTH and idx == 4: newNode = (newCell) + (WEST,)

                if   n[1] == WEST and idx == 2: newNode = (newCell) + (NORTH,)
                elif n[1] == WEST and idx == 6: newNode = (newCell) + (SOUTH,)

                if   n[1] == SOUTH and idx == 4: newNode = (newCell) + (WEST,)
                elif n[1] == SOUTH and idx == 0: newNode = (newCell) + (EAST,)

                if   n[1] == EAST and idx == 6: newNode = (newCell) + (SOUTH,)
                elif n[1] == EAST and idx == 2: newNode = (newCell) + (NORTH,)

                if idx == 8: newNode = newCell + (n[1],)

                defaultEdgeWeights = labeledEdge.LabeledEdges()

                try:
                    self.add_edge(n, newNode, weight=[idx, defaultEdgeWeights])
                except Exception, e:
                    print e

    def action(self, cell):
        return [ACT_EAST, ACT_NE, ACT_NORTH, ACT_NW, ACT_WEST]

class Car(sm.SM):
    def __init__(self, startState, world, goal):
        """
        :param startState: ((x, y), dir)
        :param world: World object
        :param goal: 2-tuple
        """
        self.startState = startState
        try:
            self.route = nx.shortest_path(world, source=startState, target=goal)
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


# run stuff
w = World(dim=1)
print w.number_of_nodes(), w.number_of_edges()
print w.neighbors((0,0,EAST))
print w.edges()
c = Car(startState=(0, 0, EAST), world=w, goal=(0, 0, SOUTH))

"""
test cases:
    1. c = Car(((1, 1), NORTH, 1), w, (0, 0)); print c.route --> should return empty list
    2. c = Car(((0, 0), NORTH, 1), w, (1, 1)); print c.route --> returns [(0, 0), (1, 1)]
"""
