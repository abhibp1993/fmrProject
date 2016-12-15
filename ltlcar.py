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
import string

# Define Operator Precedences
opPrec = {'!': 2, '&': 1, '|': 1}  # Operator Precedences
opDict = {'!': [1, lambda a: not a], '&': [2, lambda a, b: a and b], '|': [2, lambda a, b: a or b]}

# Define Directions
EAST = 0
NORTH = 1
WEST = 2
SOUTH = 3

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

    def slice(self, cell, dim, direction):
        """
        Returns the label set of square slice of size=dim based at cell along with its occupancy matrix.

        :param cell: 2-tuple (x, y)
        :param dim: integer
        :param direction: direction MACRO from {NORTH, SOUTH, EAST, WEST}
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


def slice(self, cell, dim, direction):
    # Get label
    mylabel = self.label(cell)

    # transform the world
    if direction == NORTH: numRot = 0
    elif direction == SOUTH: numRot = 2
    elif direction == EAST: numRot = 1
    else: numRot = 3

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

World.slice = slice

# TODO: Stop-Sign Behavior Add pRIORITY #3
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
        3. wait at stop sign: When the next edge is a stop sign

    Routing:
        1. A* router, when car deviates from designated route.
    """
    AVOID_OBSTACLE = 'AVOID OBSTACLE'
    GO_TO_GOAL = 'GO-TO-GOAL'
    WAIT_AT_STOP_SIGN = 'WAIT AT STOP SIGN'

    def __init__(self, world, start, goal, spec, actions, personality):
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
        self.stopSignMemory = 0
        self.personality = personality

        # LTL Processing, Automata
        self.spec = spec
        self.ltlAutomata = _automatize(spec)
        #print('Automata done... ', self.ltlAutomata.nodes(), self.ltlAutomata.edges())

        # Behaviors
        self.go2goal = Go2Goal(world=world, actions=self.actions)
        self.avoidObstacle = AvoidObstacle(world=world, actions=self.actions)
        self.waitAtStopSign = waitAtStopSign(memory=self.stopSignMemory, personality=self.personality,actions=self.actions)
        self.router = Router(world, goal, personality)

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

        :param state: 2-tuple of (label of current state, direction)
        :param inp: world file (complete world is sent, but not only slice is used further for processing)
        :return: 2-tuple of form (nState=newState, output=(action, behavior))
        """
        assert isinstance(inp, World), 'inp must be a World instance.'

        # Slice out the world (ideally this should be the input)
        x, y = inp.cell(state[0])
        if np.NaN in [x, y]:
            print('Car is out of world! Flying, eh?')
            return state, (None, 'Flying')

        visWorld = inp.slice([x-(self.visibility - 1) / 2, y], self.visibility, state[1])
        #print(np.rot90(visWorld))

        # Are we on proper route? - Get next ideal move
        suggestedMove = self.router.step(inp=state[0])      # Modify input according to router machine requirements
        # Check if obstacle is present in view
        myView = self._getViewInfo(inp, visWorld)
        obs = [myView[k].obstacle for k in myView.keys()]
        if True in obs:
            obsLoc = list(myView.keys())
            obsLoc = obsLoc[obs.index(True)]

        # # Check if stop sign is next
        nextStop = (inp.stopMap[x][y])  # TODO TEST PRIORITY 3

        # print (np.rot90(visWorld))
        # for k in myView.keys():
        #     print(k, str(myView[k]))

        # Select Behavior
        if nextStop:
            behavior = Car.WAIT_AT_STOP_SIGN
            initSuggestedMove = suggestedMove
            suggestedMove = self.waitAtStopSign.step(inp=(state, suggestedMove, self.stopSignMemory))
            suggestedMove = suggestedMove(state)[0]
            if suggestedMove == initSuggestedMove:
                nextStop = False
            else:
                self.stopSignMemory = self.stopSignMemory + 1


        if True in obs:     # NOTE NOT ELSE IF!
            behavior = Car.AVOID_OBSTACLE
            action = self.avoidObstacle.step(inp=(visWorld, state, obsLoc, suggestedMove))   # TODO ensure memory resets if action
            if action != suggestedMove: self.stopSignMemory = 0
        else:
            if not nextStop:
                behavior = Car.GO_TO_GOAL
                self.stopSignMemory = 0
            action = self.go2goal.step(inp=(state, suggestedMove))      # inp - current position, suggested step

        # Perform action
        output = action(list(inp.cell(state[0])) + [state[1]])
        nextState = [inp.label(output[0:2]), output[2]]
        #print("my memory is", self.stopSignMemory)
        # Return
        return nextState, (action, behavior, nextState)


def gnv(self, state, inp):
    assert isinstance(inp, World), 'inp must be a World instance.'

    # Slice out the world (ideally this should be the input)
    x, y = inp.cell(state[0])
    if np.NaN in [x, y]:
        print('Car is out of world! Flying, eh?')
        return state, (None, 'Flying')

    #visWorld = inp.slice([x - (self.visibility - 1) / 2, y], self.visibility, state[1])
    visWorld = inp.slice([x, y], self.visibility, state[1])
    # print(np.rot90(visWorld))

    # Are we on proper route? - Get next ideal move
    suggestedMove = self.router.step(inp=state[0])  # Modify input according to router machine requirements

    # Check if obstacle is present in view
    myView = self._getViewInfo(inp, visWorld)
    obs = [myView[k].obstacle if k != state[0] else False for k in myView.keys()]

    if True in obs:
        obsLoc = list(myView.keys())
        obsLoc = obsLoc[obs.index(True)]

    if True in obs:
        behavior = Car.AVOID_OBSTACLE
        action = self.avoidObstacle.step(inp=(visWorld, state, obsLoc, suggestedMove))  # TODO ensure memory resets if action
    else:
        behavior = Car.GO_TO_GOAL
        action = self.go2goal.step(inp=(state, suggestedMove))

    # Perform action
    output = action(list(inp.cell(state[0])) + [state[1]])
    nextState = [inp.label(output[0:2]), output[2]]
    # print("my memory is", self.stopSignMemory)
    # Return
    return nextState, (action, behavior, nextState)

Car.getNextValues = gnv

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
        (label, dir), suggestedMove = inp

        # Compute reachable set
        reachableSet = list()
        for act in self.actions:
            currCell = list(self.world.cell(label))
            outCell = act(currCell + [dir])
            outCell = (self.world.label(outCell[0:2]), outCell[2])
            reachableSet.append(outCell)

        # reachableSet = [self.world.label(act([self.world.cell(label), dir])[0:2])
        #                 for act in self.actions if act([self.world.cell(label), dir])[0:2] in self.world]

        # If suggestedMove is reachable, take it.
        if suggestedMove in [s[0] for s in reachableSet]:
            return state, self.actions[[s[0] for s in reachableSet].index(suggestedMove)]

        # Else, take the best possible move
        else:
            # Compute euclidean distances from suggestedMove
            dist = [self.world.dist(suggestedMove, s[0]) for s in reachableSet] #TODO use actual weighting

            # Find index of least distance move
            minIdx = dist.index(min(dist))

            # Make the move
            return state, self.actions[minIdx]


# TODO: Check if plans are correct.
# Route Machine
class Router(sm.SM):
    def __init__(self, world, goal, personality):
        """
        Constructs a router machine, that generates a high-level plan to reach the goal.

        :param world: World object
        :param goal: label of cell in world. (integer)
        :param personality: weighting factor for the car
        """
        # Local variables
        self.world = world
        self.goal = goal
        self.personality = personality

        # State machines variables
        self.startState = list()        # State is route/path plan as list of cells to visit - initialized with no plan

        # Graphify
        self.graph = self._graphify(world, personality)

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
            nState = nx.astar_path(G=self.graph, source=inp, target=self.goal, weight='cost')
        else:
            nState = copy.deepcopy(state)           # Avoid mutability issues

        # If length of nState is 0 or 1, there doesn't exist any path to goal.
        if len(nState) == 0 or len(nState) == 1:
            return nState, inp

        # Else compute output and modify nState and return
        output = nState[1]
        nState.pop(0)
        return nState, output

    def _graphify(self, world, personality):
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
            if [x, y] in world: grf.add_edge(label, world.label([x, y]), weight=[world.roadMap[x, y], world.grassMap[x,y], world.obsMap[x, y], world.yieldMap[x,y], world.stopMap[x,y]])            # self
            if [x, y+1] in world: grf.add_edge(label, world.label([x, y+1]), weight=[world.roadMap[x, y+1], world.grassMap[x,y+1], world.obsMap[x, y+1], world.yieldMap[x,y+1], world.stopMap[x,y+1]])       # up
            if [x, y-1] in world: grf.add_edge(label, world.label([x, y-1]), weight=[world.roadMap[x, y-1], world.grassMap[x,y-1], world.obsMap[x, y-1], world.yieldMap[x,y-1], world.stopMap[x,y-1]])        # down
            if [x+1, y] in world: grf.add_edge(label, world.label([x+1, y]), weight=[world.roadMap[x+1, y], world.grassMap[x+1,y], world.obsMap[x+1, y], world.yieldMap[x+1,y], world.stopMap[x+1,y]])        # right
            if [x-1, y] in world: grf.add_edge(label, world.label([x-1, y]), weight=[world.roadMap[x-1, y], world.grassMap[x-1,y], world.obsMap[x-1, y], world.yieldMap[x-1,y], world.stopMap[x-1,y]])        # left
            if [x+1, y+1] in world: grf.add_edge(label, world.label([x+1, y+1]), weight=[world.roadMap[x+1,y+1], world.grassMap[x+1,y+1], world.obsMap[x+1,y+1], world.yieldMap[x+1,y+1], world.stopMap[x+1,y+1]])    # up-right
            if [x+1, y-1] in world: grf.add_edge(label, world.label([x+1, y-1]), weight=[world.roadMap[x+1,y-1], world.grassMap[x+1,y-1], world.obsMap[x+1,y-1], world.yieldMap[x+1,y-1], world.stopMap[x+1,y-1]])    # down-right
            if [x-1, y+1] in world: grf.add_edge(label, world.label([x-1, y+1]), weight=[world.roadMap[x-1,y+1], world.grassMap[x-1,y+1], world.obsMap[x-1,y+1], world.yieldMap[x-1,y+1], world.stopMap[x-1,y+1]])    # up-left
            if [x-1, y-1] in world: grf.add_edge(label, world.label([x-1, y-1]), weight=[world.roadMap[x-1,y-1], world.grassMap[x-1,y-1], world.obsMap[x-1,y-1], world.yieldMap[x-1,y-1], world.stopMap[x-1,y-1]])    # down-left

        # Compute costs of all the edges
        def _getCost(weights, values):
            assert len(weights) == len(values), "Length of values not equal to length of weights. len(weight)=%d, len(values)=%d"%(len(weights), len(values))
            assert False in [i < 0 for i in values], "Values should be strictly non negative"
            #assert values[0] >= 1, "For A* to work, this weight must be at least 1"

            tempCost = 0
            for i in range(len(weights)):
                tempCost = tempCost + weights[i] * values[i]
            return tempCost

        for edg in grf.edges():
            grf[edg[0]][edg[1]]['cost'] = _getCost(personality, grf.get_edge_data(edg[0], edg[1])['weight'],)

        return grf


# waitAtStopSign
# TODO: PRIORITY 3
# if there is a stop sign, this will iterate stopsign memory and wait
class waitAtStopSign(sm.SM):
    def __init__(self, memory, personality, actions):

        # Local variables
        self.memory = memory
        self.personality = personality
        self.myActions = actions
        # Initialize the state machine
        self.initialize()

    #Based on a car's personality, the car will wait at the stop sign and return the input suggested action if the time has completed, otherwise will return the current cell as the suggestion
    def getNextValues(self, state, inp):
        if inp[2] < 5*self.personality[2]: #hasn't waited long enough for personality
            return None, self.myActions[0] #SHOULD RETURN WAIT
        else:
            return None, inp[1] #SHOULD RETURN ORIGINAL ACTION


# TODO: Opponent plays first. PRIORITY 2
# TODO: Assume knowledge dynamics of opponent. PRIORITY 4
# AvoidObstacle
class AvoidObstacle(sm.SM):
    def __init__(self, world, actions):

        # Define LTL Specification for Obstacle Avoidance
        self.ap = {'c': lambda x, y: x == y}
        self.spec = 'G!c'       # c stands for collision
        self.auto = _automatize(self.spec)

        # Local variables
        self.world = world
        self.myActions = actions
        #TODO: should not assume the obstacle can move the same way as the car, should take in dynamics of obstacle instead
        self.obsAction = [(lambda x: [x[0], x[1]]),
                          (lambda x: [x[0], x[1]+1]),
                          (lambda x: [x[0], x[1]-1]),
                          (lambda x: [x[0]+1, x[1]]),
                          (lambda x: [x[0]+1, x[1]+1]),
                          (lambda x: [x[0]+1, x[1]-1]),
                          (lambda x: [x[0]-1, x[1]]),
                          (lambda x: [x[0]-1, x[1]+1]),
                          (lambda x: [x[0]-1, x[1]-1])]

        # Initialize the state machine
        self.initialize()

    def getNextValues(self, state, inp):
        """
        State of machine is nothing. (Using memory to store last actions might prove useful, but currently ignored)
        Input to machine is observable world slice, obstacle car location and suggested next move.

        Assumptions:
            1. Obstacle car can move to any of its 8-neighbors.
            2. Game is turn-based with where our car is expected to make its move first.
            3. Our location in observable world slice is at

        :param state: None
        :param inp: 3-tuple of (world slice, myCar position, obs-car position, next suggested move)
        :return: 2-tuple of (None, action)
        """
        # Decouple input
        worldSlice, myCar, obsCar, suggestedMove = inp

        # Generate 1-step game graph
        grf = self._graphifyOneStep(myCar, obsCar)
        #print('Nodes: ', grf.number_of_nodes(), '\n', 'Edges: ', grf.number_of_edges())

        # Compute product of automata and graph
        prodAuto = self._prodAutoGraph(self.auto, grf)
        #for e in prodAuto.edges(): print(e)

        # Setup reachability game for obstacle car
        F_dash = list(set(prodAuto.nodes()) - set(prodAuto.finalStates))  # Compute the complement set of F for game
        attr, subAttr = _attractor(prodAuto, F_dash, False)
        safeStates = set(prodAuto.nodes()) - attr

        # Extract safe states for player 1 to move
        safeMoves = set()
        for s in safeStates:
            safeMoves.add(s[0][0][0])

        # Compute reachable set
        reachableSet = list()
        for act in self.myActions:
            currCell = list(self.world.cell(myCar[0]))
            outCell = act(currCell + [myCar[1]])
            outCell = (self.world.label(outCell[0:2]), outCell[2])
            reachableSet.append((outCell, act))

        # Compute possible moves as intersection of reachable set and safe moves
        possibleMoves = list(set([k[0] for k in reachableSet]) & safeMoves)

        # Check if suggested move is possible
        if suggestedMove in [p[0] for p in possibleMoves]:
            return None, reachableSet[[p[0] for p in possibleMoves].index(suggestedMove)][1]
        else:
            return None, self.myActions[0]        # Need to write selection method. (dummy for now) # TODO: PRIORITY 5

    def _graphifyOneStep(self, p1, p2):
        """
        Nodes of graph are formatted as ((p1:int, p2:int), turnOfP1:bool)
        Assumption 1: self.world, self.myActions, self.obsAction are configured properly.

        :param p1: label of cell of player 1 = my SDC
        :param p2: label of cell of player 2 = obstacle car
        :return: networkx.DiGraph instance
        """
        p1, dir = p1

        # Initialize graph, start state
        grf = nx.DiGraph()
        startState = ((p1, p2), True)
        grf.startState = startState

        # Player 1 takes one step
        # reachableSet1 = [self.world.label(act(self.world.cell(p1)))
        #                 for act in self.myActions if act(self.world.cell(p1)) in self.world]

        # Compute reachable set
        reachableSet1 = list()
        for act in self.myActions:
            currCell = list(self.world.cell(p1))
            outCell = act(currCell + [dir])
            outCell = (self.world.label(outCell[0:2]), outCell[2])
            reachableSet1.append(outCell)


        frontier = set()
        for s in reachableSet1:
            newState = ((s, p2), False)
            grf.add_node(newState)
            grf.add_edge(startState, newState)
            frontier.add(newState)

        # Player 2 makes a move
        reachableSet2 = [self.world.label(act(self.world.cell(p2)))
                        for act in self.obsAction if act(self.world.cell(p2)) in self.world]

        for n in frontier:
            for s2 in reachableSet2:
                newState = ((n[0][0], s2), True)
                grf.add_node(newState)
                grf.add_edge(n, newState)

        # Return graph
        return grf

    def _prodAutoGraph(self, automata, graph, verbose=False):
        """
        Constructs the product automata of graphAutomata and gameGraph for
        turn-based zero-sum (?) game.

        Approach is incremental and exhaustive construction, i.e. all possible
        states are enumerated, with only valid transitions.

        :return: GraphAutomata object.
        """
        if verbose:
            print()
            print('Constructing Product Automata -----------------')


        # Define product automata as graph-automata
        prodAutomata = GraphAutomata()

        # Initialize start node for the product.
        prodAutomata.startState = (graph.startState, automata.startState)
        if verbose: print('\t', 'Start State of prod-Automata: ', prodAutomata.startState)

        # Initialize frontier as queue
        frontier = [prodAutomata.startState]    # To be used as Queue

        # Loop until frontier is not empty
        while len(frontier) > 0:
            # Pop a node off frontier
            expNode = frontier.pop(0)
            gameNode, autoNode = expNode

            # Get all edges from node of game-graph
            gameEdges = graph.edges(gameNode)

            # Get all edges from node of automata
            autoEdges = automata.edges(autoNode)

            # Loop over each edge of game-graph
            for gEdge in gameEdges:
                # Get destination node
                gDestNode = gEdge[1]
                p1State, p2State = gDestNode[0]

                # Compute atomic proposition values for node.
                nodeLabel = dict()
                for prop in self.ap:
                    nodeLabel[prop] = self.ap[prop](p1State, p2State)

                # For each edge in automata-graph do
                for aEdge in autoEdges:
                    # Evaluate the label of automata (of LTL) edge over this AP set.
                    aDestNode = aEdge[1]
                    aEdgeLabel = automata.labelOfEdge(aEdge[0], aEdge[1])
                    parseLabel = parseFormula(aEdgeLabel, opPrec=opPrec)
                    value = evaluateFormula(parseLabel, opDict, nodeLabel)

                    # If true, then take transition and create a state in product-automata
                    if value is True:
                        newNode = (gDestNode, aDestNode)

                        # If node doesn't exists in product-automata
                        if newNode not in prodAutomata.nodes():
                            # Then add it to frontier.
                            prodAutomata.add_node(newNode)
                            frontier.append(newNode)

                            # Add edge to prodAutomata. (Note: networkx takes care of repeatition)
                            prodAutomata.add_edge(expNode, newNode, gDestNode[1])

                            # If node is final state (accepting), add it accordingly
                            if aDestNode in automata.finalStates and newNode not in prodAutomata.finalStates:
                                prodAutomata.finalStates.append(newNode)

        if verbose:
            print('\t', 'Product Automata Nodes: ', prodAutomata.number_of_nodes(), ' Edges: ', prodAutomata.number_of_edges())
            print('\t', 'Product Automata {} Final States'.format(len(prodAutomata.finalStates)), prodAutomata.finalStates)
            print()

        #print('Duplication Check: ', len(prodAutomata.finalStates), len(set(prodAutomata.finalStates)))
        return prodAutomata


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

    # Patch 1: Avoid duplication in final states set
    grfAutomata.finalStates = list(set(grfAutomata.finalStates))

    if verbose:
        print('\t', 'graphAutomata Nodes: ', grfAutomata.nodes())
        print('\t', 'graphAutomata Edges: ', grfAutomata.edges())
        print()

    # Return graph-Automata
    return grfAutomata


# Attractor sets computation
def _attractor(graph, F, isPlayer1=True):
    # Initialize attractor
    attractor = set(F)          # stores net attractor set
    subAttr = [attractor]       # Sequentially stores n'th attractor set

    # Loop
    frontier = set()
    while True:
        # Update frontier
        lastAttr = subAttr[-1]
        for nd in lastAttr:
            inEdges = graph.in_edges(nd)
            frontier |= set([edg[0] for edg in inEdges])

        #print(frontier)

        # Iterate over nodes in frontier and find whether to make them permanant
        newAttr = set()
        for nd in frontier:
            # Decouple information from node
            (((p1, p2), turnOfPlayer1), autState) = nd

            if (turnOfPlayer1 and isPlayer1) or (not turnOfPlayer1 and not isPlayer1):
                newAttr.add(nd)

            else:
                for edge in graph.edges(nd):
                    dest = edge[1]
                    if dest not in attractor:
                        continue

                newAttr.add(nd)

        # Check loop termination condition
        if newAttr == lastAttr:
            break

        # Update attractor set
        attractor |= newAttr
        subAttr.append(newAttr)

    return attractor, subAttr


def parseFormula(formula, opPrec):
    """
    Prefix conversion required.

    :param formula: string of propositional logic formula
    @:param opDict: operator dictionary with sym: function_pointer pairs.
    :return:
    """

    # Prefix conversion helper function
    def myPrefix(formula):
        """
        Converts formula string to prefix.

        :param formula: string
        :return: list in prefix format
        """

        prefixList = list()
        operatorStack = list()
        for token in formula:
            if token in string.ascii_lowercase:
                prefixList.append(token)

            elif token in '1':
                prefixList.append(True)

            elif token in '0':
                prefixList.append(False)

            elif token in ' ':
                continue

            else:
                while (len(operatorStack) > 0) and (opPrec[operatorStack[-1]] > opPrec[token]):
                    prefixList.append(operatorStack.pop())

                operatorStack.append(token)

        while len(operatorStack) > 0:
            prefixList.append(operatorStack.pop())

        prefixList.reverse()
        return prefixList

    # Convert formula to prefix
    prefFormula = myPrefix(formula=formula)

    return  prefFormula


def evaluateFormula(formula, opDict, apDict):
    # Replace all atomic propositions in formula by values from apDict
    for i in range(len(formula)):
        if formula[i] in apDict.keys():
            formula[i] = apDict[formula[i]]

    # Loop until formula is reduced to single literal
    while len(formula) != 1:
        # Fetch last occuring operator
        myOperator = None
        idxOperator = None
        for i in range(len(formula) - 1, -1, -1):
            if formula[i] in opDict.keys():
                myOperator = formula[i]
                idxOperator = i
                break

        # Fetch number of operands required for operation from opDict
        numOperands, fcn = opDict[myOperator]

        # Reduce formula by performing operation
        if numOperands == 1:
            # Find operand
            operand = formula[idxOperator+1]

            # Complete Operation
            result = fcn(operand)

            # Replace operator and operand with result
            formula[idxOperator] = result
            formula.pop(idxOperator+1)

        elif numOperands == 2:
            # Find operands
            operand1 = formula[idxOperator + 1]
            operand2 = formula[idxOperator + 2]

            # Complete Operation
            result = fcn(operand1, operand2)

            # Replace operator and operand with result
            formula[idxOperator] = result
            formula.pop(idxOperator + 1)
            formula.pop(idxOperator + 1)

        else:
            raise Exception('Unable to process parsedExpression.')

    # Return final value of formula
    return formula[0]


# TODO: Rigorous Testing!!!
if __name__ == '__main__':
    # actions = [(lambda x: tuple([x[0] + 1, x[1]])),  # Right
    #            (lambda x: tuple([x[0], x[1] + 1])),  # Up
    #            (lambda x: tuple([x[0] - 1, x[1]])),  # Left
    #            (lambda x: tuple([x[0], x[1] - 1]))]  # Down

    # Define actions
    actions = list()
    direction = [1, 0, -1, 0]
    cos = lambda x: direction[x]
    sin = lambda x: direction[(3 + x) % 4]
    cw = lambda x: (x - 1) % 4
    ccw = lambda x: (x + 1) % 4

    def wait(cell):
        return cell                                                              # wait

    def forward(cell):
        return cell[0] + cos(cell[2]), cell[1] + sin(cell[2]), cell[2]                                # forward

    def fwdRight(cell):
        return cell[0] + sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) - cos(cell[2]), cell[2] # fr

    def fwdLeft(cell):
        return cell[0] - sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) + cos(cell[2]), cell[2]  # fl

    def right(cell):
        return cell[0] + sin(cell[2]), cell[1] - cos(cell[2]), cw(cell[2])  # RIGHT

    def left(cell):
        return cell[0] - sin(cell[2]), cell[1] + cos(cell[2]), ccw(cell[2])  # LEFT

    actions = [wait, forward, fwdRight, fwdLeft, right, left]

    # Create world
    w = World(roadMap='world55/road.bmp', dim=5, grassMap='world55/grass.bmp', stopMap='world55/stopsign.bmp')
    w.obsMap[0, 1] = 1
    print('---Obs---', '\n', np.rot90(w.obsMap))

    c = Car(start=(1, NORTH), goal=23, spec='Ga & Fb', actions=actions, world=w, personality=[0, 2, 1, 0, 1])
    #print(c.transduce([w, w, w, w, w, w]))

    #image creation portion
    img = Image.new( 'RGB', (w.dim,w.dim), "white") # create a new black image
    pixels = img.load() # create the pixel map
    #find original plan
    c.transduce([w])
    originalPath = c.router.currState
    print(originalPath)

    for x in range(0,w.dim):
        for y in range(0,w.dim):
            pixels[x,y] = (0,0,0)
    for act in range(0, len(originalPath)):
        pos = w.cell(originalPath[act])
        pixels[pos[1],pos[0]] = (0, 0, 255) # set the colour accordingly
    #solve actual
    actualPath = c.transduce([w, w, w, w, w,w,w,w,w,w,w,w,w,w,w,w,w,w,w, w,w,w,w,w,w,w,w,w,w,w,w,w,w,w])
    print(actualPath)
    #modify image
    for act in range(0, len(actualPath)):
        pos = w.cell(actualPath[act][2][0])
        pixels[pos[1],pos[0]] = (255,0,pixels[pos[1],pos[0]][2]) # set the colour accordingly
    img = img.resize((500,500))
    img.rotate(270).show()


    #r = Router(w, 0, [1,1,1,1,1])
    # print(r.transduce([0, 1, 2, 3, 3]))

