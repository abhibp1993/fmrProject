"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 3, 2016

Implements LTL-based car.
"""

import sm, spot
import numpy as np
from PIL import Image
import networkx as nx
import copy


class NodeAP:
    """
    Structure for labeling edges.
    Author: Ari, Yicong
    """
    def __init__(self, yieldSign=False, stopSign=False, illegal=False, obstacle=False, road=False, grass=False):
        self.yieldSign = yieldSign
        self.stopSign = stopSign
        self.illegal = illegal
        self.obstacle = obstacle
        self.road = road
        self.grass = grass

    @property
    def values(self):
        return [bool(self.yieldSign), bool(self.stopSign), bool(self.illegal),
                bool(self.obstacle), bool(self.road), bool(self.grass)]

    def __str__(self):
        return 'grass: ' + str(self.grass) + \
               ' road: ' + str(self.road) + \
               ' obs: ' + str(self.obstacle) + \
               ' stop: ' + str(self.stopSign) + \
               ' yield: ' + str(self.yieldSign)


# World
class World(object):
    def __init__(self, roadMap, dim=10, obsMap=None, yieldMap=None, stopMap=None, grassMap=None):
        # Store local variables
        self.dim = dim          # Dimension of square world

        # Generate Label Map
        self.labelMap = self._generateLabels(dim)    # Generates labels to dim * dim world

        # Generate empty bitmaps
        self.roadMap = np.zeros((dim, dim), dtype=bool)
        self.grassMap = np.ones((dim, dim), dtype=bool)
        self.obsMap = np.zeros((dim, dim), dtype=bool)
        self.yieldMap = np.zeros((dim, dim), dtype=bool)
        self.stopMap = np.zeros((dim, dim), dtype=bool)

        # If bitmaps are provided, attempt parsing them.
        if roadMap != None:
            self.roadMap = self._bmpParser(roadMap)
        if grassMap != None:
            self.grassMap = self._bmpParser(grassMap)
        if obsMap != None:
            self.obsMap = self._bmpParser(obsMap)
        if yieldMap != None:
            self.yieldMap = self._bmpParser(yieldMap)
        if stopMap != None:
            self.stopMap = self._bmpParser(stopMap)

        # Validation
        try:
            self._validateWorld()
        except AssertionError:
            print('World could not be instantiated. Check input bitmaps for consistency.')

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.labelMap
        elif isinstance(item, (tuple, list)):
            return self.label(item) in self.labelMap
        else:
            return False


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

    def slice(self, cell, dim):
        """
        Returns the label set of square slice of size=dim based at cell along with its occupancy matrix.

        :param cell: 2-tuple (x, y)
        :param dim: integer
        :return: 2-tuple (labelSet, obsSet)
        @remark: Take care of np.NaNs properly while using it.
        """
        assert cell[0] + dim > 0 and cell[1] + dim > 0 \
            and cell[0] < self.dim and cell[1] < self.dim, 'World.slice: View is out of bound'

        # Get base X, Y positions
        cell_x = cell[0]
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
        slice = np.zeros((dim, dim))
        slice[:] = np.NaN

        # Update the slice
        slice[slice_x_min:slice_x_max, slice_y_min:slice_y_max] = \
            self.labelMap[world_x_min:world_x_max, world_y_min:world_y_max]

        # Return
        return slice

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


# Car
class Car(sm.SM):
    """
    Defines a Self-Driving car with following assumptions:
        1. At any given moment, no more than 1 obstacle appears in the view of car.
            This keeps state-explosion under control!
        2. The obstacle cannot move more than 1 step in any given time-step.

    Behaviors:
        1. Go-to-Goal: Default behavior when there is no obstacle in view.
        2. Avoid Obstacle: When there is some obstacle in view.

    Routing:
        1. A* router, when car deviates from designated route.
    """
    AVOID_OBSTACLE = 'AVOID OBSTACLE'
    GO_TO_GOAL = 'GO-TO-GOAL'

    def __init__(self, world, start, goal, spec, actions):
        """

        :param start: label of cell. Integer
        :param goal: label of cell. Integer
        :param spec: LTL string, compatible with spot module.
        :param actions: list of functions of template: newCell <-- fcn(cell)
        """
        # Local variables
        self.startState = start
        self.goal = goal
        self.actions = actions
        self.visibility = 3

        # LTL Processing, Automata
        self.spec = spec
        self.ltlAutomata = _automatize(spec)
        print('Automata done... ', self.ltlAutomata.nodes(), self.ltlAutomata.edges())

        # Behaviors
        self.go2goal = Go2Goal(world=world, actions=self.actions)
        self.avoidObstacle = AvoidObstacle()
        self.router = Router(world, goal)

        # Initialize state machine
        self.initialize()

    def _getViewInfo(self, world, slice):
        """
        Generates a labeled map for all cells in the slice of world.

        :param world: World instance
        :param slice: labels of cells from world.
        :type slice: 2D np.array
        :return: dictionary of {label: NodeAP object}
        """
        # Initialize the dictionary
        lMapDict = dict()

        # Iteratively populate dictionary
        for label in np.nditer(slice):
            if not np.isnan(label):
                lMapDict.update({int(label): world.info(int(label))})

        return lMapDict

    def getNextValues(self, state, inp):
        """
        Defines the transition function for self-driving car.

        :param state: label of current state
        :param inp: world file (complete world is sent, but not only slice is used further for processing)
        :return: 2-tuple of form (nState=newState, output=(action, behavior))
        """
        assert isinstance(inp, World), 'inp must be a World instance.'

        # Slice out the world (ideally this should be the input)
        x, y = inp.cell(state)
        if np.NaN in [x, y]:
            print('Car is out of world! Flying, eh?')
            return state, (None, 'Flying')

        visWorld = inp.slice([x-(self.visibility - 1) / 2, y], self.visibility)
        #print(np.rot90(visWorld))

        # Are we on proper route? - Get next ideal move
        suggestedMove = self.router.step(inp=state)      # Modify input according to router machine requirements

        # Check if obstacle is present in view
        myView = self._getViewInfo(inp, visWorld)
        obs = True in [myView[k].obstacle for k in myView.keys()]

        # print (np.rot90(visWorld))
        # for k in myView.keys():
        #     print(k, str(myView[k]))

        # Select Behavior
        if obs:
            behavior = Car.AVOID_OBSTACLE
            action = self.avoidObstacle.step(inp=None)       # inp- suggested step
        else:
            behavior = Car.GO_TO_GOAL
            action = self.go2goal.step(inp=(state, suggestedMove))             # Pass appropriate input

        # Perform action
        nextState = inp.label(action(inp.cell(state)))

        # Return
        return nextState, (action, behavior)


# GoToGoal
class Go2Goal(sm.SM):
    """
    Represents a memoryless state machine.
    The next action is chosen based on greedy strategy as described below.
        1. compute all reachable states from current state
        2. if suggestedState is in reachable set, choose it.
        3. else choose least-deviation inducing transition..
    """
    def __init__(self, world, actions):
        self.startState = None
        self.actions = actions
        self.world = world

        self.initialize()

    def getNextValues(self, state, inp):
        """
        See class description.

        :param state: None
        :param inp: 2-tuple of (currState, suggestedMove), currState, suggestedMove - label of cell
        :return: 2-tuple of (nextState, action)
        """
        # Copy information
        currState, suggestedMove = inp

        # Compute reachable set
        reachableSet = [self.world.label(act(self.world.cell(currState)))
                        for act in self.actions if act(self.world.cell(currState)) in self.world]

        # If suggestedMove is reachable, take it.
        if suggestedMove in reachableSet:
            return state, self.actions[reachableSet.index(suggestedMove)]

        # Else, take the best possible move
        else:
            # Compute euclidean distances from suggestedMove
            dist = [self.world.dist(suggestedMove, l) for l in reachableSet]

            # Find index of least distance move
            minIdx = dist.index(min(dist))

            # Make the move
            return state, self.actions[minIdx]





# Route Machine
class Router(sm.SM):
    def __init__(self, world, goal):
        """
        Constructs a router machine, that generates a high-level plan to reach the goal.

        :param world: World object
        :param goal: label of cell in world. (integer)
        """
        # Local variables
        self.world = world
        self.goal = goal

        # State machines variables
        self.startState = list()        # State is route/path plan as list of cells to visit - initialized with no plan

        # Graphify
        self.graph = self._graphify(world)

        # Initialize State Machine
        self.initialize()

    def getNextValues(self, state, inp):
        """
        State is last known path. The transition is defined by
            1. If current state is in the last computed path, return next move from that path plan.
            2. If car has deviated, recompute the plan, and suggest next move.

        Output is next suggested cell (as label).

        A*/dijkstra implementation is expected.

        :param state: last route as list of nodes in self.graph
        :param inp: current cell label (Integer)
        :return: next suggested cell label (Integer)
        """
        # If route is not previously computed or car has deviated off, reroute.
        if len(state) == 0 or inp not in state:
            nState = nx.astar_path(G=self.graph, source=inp, target=self.goal)
        else:
            nState = copy.deepcopy(state)           # Avoid mutability issues

        # If length of nState is 0 or 1, there doesn't exist any path to goal.
        if len(nState) == 0 or len(nState) == 1:
            return nState, inp

        # Else compute output and modify nState and return
        output = nState[1]
        nState.pop(0)
        return nState, output

    def _graphify(self, world):
        """
        Generates default 8-connectivity between all cells of world.
        This is trivial function. Needs to be refined to appear realistic.

        :param world: world instance
        :return: networkx.graph
        """
        assert isinstance(world, World), 'Router._graphify: world must be a World instance.'

        # Initialize a graph
        grf = nx.DiGraph()

        # For each cell label in world, create a node.
        for label in np.nditer(world.labelMap):
            grf.add_node(int(label))

        # For each cell, compute 8-neighbors and add edges to graph. (Take care of cells falling out of world)
        for label in np.nditer(world.labelMap):
            label = int(label)
            x, y = world.cell(label)

            if [x, y] in world: grf.add_edge(label, world.label([x, y]))            # self
            if [x, y+1] in world: grf.add_edge(label, world.label([x, y+1]))        # up
            if [x, y-1] in world: grf.add_edge(label, world.label([x, y-1]))        # down
            if [x+1, y] in world: grf.add_edge(label, world.label([x+1, y]))        # right
            if [x-1, y] in world: grf.add_edge(label, world.label([x-1, y]))        # left
            if [x+1, y+1] in world: grf.add_edge(label, world.label([x+1, y+1]))    # up-right
            if [x+1, y-1] in world: grf.add_edge(label, world.label([x+1, y-1]))    # down-right
            if [x-1, y+1] in world: grf.add_edge(label, world.label([x-1, y+1]))    # up-left
            if [x-1, y-1] in world: grf.add_edge(label, world.label([x-1, y-1]))    # down-left

        return grf


# AvoidObstacle
class AvoidObstacle(sm.SM):
    def __init__(self):
        self.initialize()

    def getNextValues(self, state, inp):
        pass


# Helper Class Automata: Represents Labeled transitions, states
class GraphAutomata(nx.DiGraph):
    def __init__(self):
        super(GraphAutomata, self).__init__()
        self.startState = None
        self.finalStates = list()
        self.atomicPropositions = None

    def add_edge(self, src, dest, label):
        super(GraphAutomata, self).add_edge(src, dest)
        self[src][dest]['label'] = label

    def labelOfEdge(self, src, dest):
        try:
            return self[src][dest]['label']
        except KeyError:
            return ''

    def printNodes(self):
        myStr = 'Nodes: '
        for n in self.nodes():
            myStr += str(n) + ','

        print(myStr[0:-1])


# Automatization of LTL Specification
def _automatize(spec, verbose=False):
    """
    Parses the LTL task-specification into an automata accepting its language.
    The automata (in spot-module's format) is converted to Automata class object
    which is easy to use in our framework.

    :param spec: string LTL formula
    :param verbose: True if progress is to be printed. False otherwise.
    :return: GraphAutomata object
    """
    if verbose:
        print()
        print('Constructing LTL language accepting automata --------------')

    # Helper function for printing spot-automata (customized from spot-documentation)
    def spot_automata_print(aut):
        bdict = aut.get_dict()
        print('\t', "Number of states: ", aut.num_states())
        print('\t', "Initial states: ", aut.get_init_state_number())
        print('\t', "Atomic propositions:", end='')
        for ap in aut.ap():
            print('\t', ' ', ap, ' (=', bdict.varnum(ap), ')', sep='', end='')
        print()
        print('\t', "Deterministic:", aut.prop_deterministic())
        print('\t', "State-Based Acc:", aut.prop_state_acc())
        for s in range(0, aut.num_states()):
            print('\t', "State {}:".format(s))
            for t in aut.out(s):
                print('\t', "  edge({} -> {})".format(t.src, t.dst))
                print('\t', "    label =", spot.bdd_format_formula(bdict, t.cond))
                print('\t', "    acc sets =", t.acc)

    # Create Buchi Automata using spot module for basic LTL parsing.
    spotAutomata = spot.translate(spec, 'BA', 'complete')
    if verbose:
        print()
        print('\t', '...............Printing spot automata model...............')
        spot_automata_print(spotAutomata)
        print('\t', '...............')

    # Define graph-Automata
    grfAutomata = GraphAutomata()

    # Extract basic information from spot-Automata
    bdict = spotAutomata.get_dict()
    grfAutomata.startState = spotAutomata.get_init_state_number()
    grfAutomata.atomicPropositions = spotAutomata.ap()
    import re

    if verbose:
        print()
        print('\t', 'Atomic Propositions captured: ', grfAutomata.atomicPropositions)
        print('\t', 'Start State captured: ', grfAutomata.startState)

    # Add all edges with respective labels
    for src in range(spotAutomata.num_states()):
        s = int(src)  # Node(name=str(src))
        grfAutomata.add_node(s)

        for dest in spotAutomata.out(src):
            d = int(dest.dst)
            grfAutomata.add_node(d)
            grfAutomata.add_edge(s, d, str(spot.bdd_format_formula(bdict, dest.cond)))

            finalStateSet = re.findall(r'\d+', str(dest.acc))  # The logic for this is not clear.
            if len(finalStateSet) > 0: grfAutomata.finalStates.append(s)  # This appeared to best explanation!

    if verbose:
        print('\t', 'graphAutomata Nodes: ', grfAutomata.nodes())
        print('\t', 'graphAutomata Edges: ', grfAutomata.edges())
        print()

    # Return graph-Automata
    return grfAutomata


def testWorld():
    # Create world
    w = World(roadMap='road.bmp', dim=5, grassMap='grass.bmp')
    print('--Labels--', '\n', np.rot90(w.labelMap))  # Rotation is for visual convenience
    print('--Grass--', '\n', np.rot90(w.grassMap))
    # print(w.cell(0), w.cell(28))    # Gives cell given label.

    # Slice world
    # try:
    #     print(np.rot90(w.slice((-3, -3), 3)))       # Case1: Out of bounds on lower-left corner
    # except:
    #     print('Lower Left Out of Bounds.')
    #
    # try:
    #     print(np.rot90(w.slice((5, 5), 3)))         # Case2: Out of bounds on upper right corner
    # except:
    #     print('Upper Right Out of Bounds.')

    # print(np.rot90(w.slice((-2, -2), 3)))           # Case3: Left corner partial view
    # print(np.rot90(w.slice((0, 0), 3)))             # Case4: Full view
    # print(np.rot90(w.slice((3, 3), 3)))             # Case3: Right corner partial view

    # Test BMP Parser
    # print(w._bmpParser('test.bmp'))

    # Extract info
    print(w.info(20))


if __name__ == '__main__':
    #testWorld()
    actions = [(lambda x: tuple([x[0] + 1, x[1]])),  # Right
               (lambda x: tuple([x[0], x[1] + 1])),  # Up
               (lambda x: tuple([x[0] - 1, x[1]])),  # Left
               (lambda x: tuple([x[0], x[1] - 1]))]  # Down


    w = World(roadMap='road.bmp', dim=5, grassMap='grass.bmp')
    c = Car(start=1, goal=23, spec='Ga & Fb', actions=actions, world=w)
    print(c.transduce([w, w, w, w]))

    r = Router(w, 0)
    # print(r.transduce([0, 1, 2, 3, 3]))


