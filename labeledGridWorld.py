"""
Author: Ari Goodman, Abhishek Kulkarni
Modified: November 30, 2016

Creates labeled grid with labels for nodes and edges and uses A* to solve.
"""

import networkx as nx
import sm
import labeledEdge
from PIL import Image
import numpy as np

EAST = 0
NORTH = 1
WEST = 2
SOUTH = 3


class World(nx.DiGraph):
    def __init__(self, dim=10):
        super(World, self).__init__()
        self.dim = dim
        self.grid = self._createGrid()
        self._graphify()

    def __contains__(self, cell):
        cells = [i for i in self.nodes()]
        return cell in cells

    def _createGrid(self):
        grid = list()
        for i in range(self.dim):
            for j in range(self.dim):
                grid.append((i, j))
        return grid

    def _bmpToFlags(self, bmp_data, flag):
        # flag: indicates which property of the land we are going to assign.
        # bmp_data: indicates
        assert len(bmp_data) == self.dim, "size is incorrect"
        assert len(bmp_data[0]) == self.dim, "size is incorrect"
        for edge in self.edges():
            endNode = edge[1]
            weights = self.get_edge_data(edge[0], edge[1])["weight"]
            weights[flag] = bool(bmp_data[endNode[0]][endNode[1]])
            self[edge[0]][edge[1]]["data"] = weights

    def _graphify(self):
        # Generate all nodes
        for cell in self.grid:
            self.add_node(cell + (NORTH,))
            self.add_node(cell + (EAST,))
            self.add_node(cell + (WEST,))
            self.add_node(cell + (SOUTH,))

        # List all possible action
        actions = list()
        direction = [1, 0, -1, 0]
        cos = lambda x: direction[x]
        sin = lambda x: direction[(3 + x) % 4]

        cw = lambda x: (x - 1) % 4
        ccw = lambda x: (x + 1) % 4

        actions.append(lambda cell: (cell[0], cell[1], cell[2]))  # wait
        actions.append(lambda cell: (cell[0] + cos(cell[2]), cell[1] + sin(cell[2]), cell[2]))  # forward
        actions.append(
            lambda cell: (cell[0] + sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) - cos(cell[2]), cell[2]))  # fr
        actions.append(
            lambda cell: (cell[0] - sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) + cos(cell[2]), cell[2]))  # fl
        actions.append(lambda cell: (cell[0] + sin(cell[2]), cell[1] - cos(cell[2]), cw(cell[2])))  # RIGHT
        actions.append(lambda cell: (cell[0] - sin(cell[2]), cell[1] + cos(cell[2]), ccw(cell[2])))  # LEFT
        costs = [1, 2, 3, 3, 4, 5]
        # Perform each action on node, and add new edges
        for n in self.nodes():
            for act in actions:
                newCell = act(n)
                if newCell not in self:
                    continue

                newNode = newCell
                defaultEdgeWeights = labeledEdge.LabeledEdges()

                try:
                    self.add_edge(n, newNode, weight=[costs[actions.index(act)]] + defaultEdgeWeights.values)
                except Exception, e:
                    print e

    def extractBmpData(self, bmp):
        im = Image.open(bmp, 'r')
        pix_val = list(im.getdata())
        for pix in range(0, len(pix_val)):
            if pix_val[pix] > 0:
                pix_val[pix] = 0
            else:
                pix_val[pix] = 1
        print pix_val
        return np.reshape(np.array(pix_val), im.size)


class Car(sm.SM):
    def __init__(self, startState, world, goal, values):
        """
        :param startState: ((x, y), dir)
        :param world: World object
        :param goal: 2-tuple
        """
        self.startState = startState
        for edge in world.edges():
            cost = self.getCost(world.get_edge_data(edge[0], edge[1])["weight"], values)
            world[edge[0]][edge[1]]["data"] = cost
        try:
            self.route = nx.astar_path(world, startState, goal,
                                       weight='data')  # nx.shortest_path(world, source=startState, target=goal)
        except Exception, e:
            print e
            self.route = list()
        print self.route

    def getCost(self, weights, values):
        assert len(weights) == len(values), "Length of values not equal to length of weights"
        assert False in [i < 0 for i in values], "Values should be strictly non negative"
        assert values[0] >= 1, "For A* to work, this weight must be atleast 1"
        tempCost = 0
        for i in range(len(weights)):
            tempCost = tempCost + weights[i] * values[i]
        return tempCost

    def getNextValues(self, state, inp):
        """
        :param state:
        :param inp:
        :return:
        """
        # Get next action
        nextDesiredAction = self.route.pop(0)


# run stuff
w = World(dim=5)
# w._bmpToFlags([[0, 1, 0, 1, 1],[0, 1, 0, 0, 0],[0, 1, 0,1 ,1],[0, 1, 0,1 ,1],[0, 1, 0,1 ,1]],3)
# c = Car(startState=(1, 1, NORTH), world=w, goal=(4, 0, SOUTH), values=[1]*7)

# Rembmer: for the bmp, 0 is black, 1 is white
w._bmpToFlags(w.extractBmpData('road.bmp'), 4)

for edge in w.edges():
    print w.get_edge_data(edge[0],edge[1])["weight"]
