"""
Author: Abhishek Kulkarni
Modified: November 14, 2016

Implements basic data structures world creation
"""

import numpy.linalg as linalg
import heapq


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def __len__(self):
        return len(self._queue)


class Node(object):
    EXPLORED = 1
    FRONTIER = 0
    UNEXPLORED = -1

    def __init__(self, name, cell):
        self.name = name
        self.pathLength = float('inf')
        self.parent = None
        self.explored = Node.UNEXPLORED
        self.heuristicVal = 0
        self.cell = cell

    def heuristic(self, goal):
        """
        Defines euclidean distance heuristic value for node.
        @param goal: 2-tuple (i, j)
        """
        return linalg.norm([self.cell[0] - goal.cell[0], self.cell[1] - goal.cell[1]])

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name


class Edge(object):
    ACTIVE = 'active'
    ISCLOSING = 'isclosing'

    def __init__(self, src, dest, cells, weight=0, atIntersection=False):
        self.src = src
        self.dest = dest
        self.cells = cells
        self.ap = {Edge.ACTIVE: True, Edge.ISCLOSING: False}
        self.isInter = atIntersection
        self.weight = weight

    def cost(self, aggression=None):
        """
        Returns a effective-cost of edge, based on car aggression.
        Note: Being aggressive doesn't change anything for a normal road edge. But
        makes a difference for an edge at intersection.

        @TODO: Implement more sophisticated function cost calculation.
        """
        return len(self.cells) + self.weight

    def __str__(self):
        return str(self.src) + ' -' + str(self.cost()) + '-> ' + str(self.dest)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.src == other.src and self.dest == other.dest


class Graph(object):
    def __init__(self, V, E):

        self.G = dict()
        self.E = E

        for v in V:
            self.G[v] = list()

        for e in E:
            self.G[e.src].append(e)

    def getChildEdges(self, n):
        try:
            eList = self.G[n]
            return eList
        except:
            return []

    def getChildNodes(self, n):

        children = []
        try:
            for e in self.G[n]:
                children.append(e.dest)

            return children
        except:
            return children

    def getEdge(self, n1, n2):
        eList = self.G[n1]
        for e in eList:
            if e.dest == n2:
                return e

        return None

    def __str__(self):
        return str([str(n) for n in self.G.keys()]) + ', ' + str([str(e) for e in self.E])


def aStar(G, src, dest, aggression=0):
    # set startNode's Attributes
    src.parent = None
    src.pathLength = 0

    # initialize frontier
    frontier = PriorityQueue()
    frontier.push(src, src.pathLength + src.heuristic(dest))

    while (len(frontier) != 0):
        tmpNode = frontier.pop()
        tmpNode.explored = Node.EXPLORED

        if tmpNode == dest:
            path = []
            n = dest
            while not (n is None):  # Dicey -> What's the guarantee that n.getParent = None happens only at StartNode??
                path.append(n)
                n = n.parent
            path.reverse()
            return path

        children = G.getChildEdges(tmpNode)
        for child in children:
            bestLength = tmpNode.pathLength + child.cost(aggression)
            if bestLength < child.dest.pathLength:
                child.dest.parent = tmpNode
                child.dest.pathLength = bestLength
                frontier.push(child.dest, bestLength)


def edgify(G, nList):
    eList = []

    if len(nList) < 2: return []
    for i in range(len(nList)-1):
        eList.append(G.getEdge(nList[i], nList[i+1]))

    return eList


n1 = Node('n1', [1, 1])
n2 = Node('n2', [2, 2])
V = [n1, n2]

e = Edge(n1, n2, [[1, 1], [1, 2], [2, 2], [2, 3]])
E = [e]

G = Graph(V, E)

path = aStar(G, n1, n2)
edge = edgify(G, path)
for e in edge: print e