#!/usr/bin/env python
"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 16, 2016

Defines car with discrete dynamics for simulation.
"""

# Imports
import sm
import networkx as nx
import numpy as np
import simworld


# Define macros
NORTH = 90
WEST = 180
SOUTH = 270
EAST = 0


# TODO: Make the operations efficient. There's a lot of scope for improvement!
def fwd(world, pose):
    """
    Returns new pose after performing forward operation on pose.

    :param world: simworld.World object
    :param pose: 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    :return: pose 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    """
    # Decouple arguments and validate
    # TODO: Verify if assertion errors works as expected.
    cell_label, heading = pose
    assert cell_label in world, 'opertion fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'opertion fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {NORTH: (1, 0), EAST: (0, 1), SOUTH: (-1, 0), WEST: (0, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return None


def right(world, pose):
    """
    Returns new pose after performing right operation on pose.
    Definition of right is - moves to right cell and turns by -90 deg.

    :param world: simworld.World object
    :param pose: 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    :return: pose 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    """
    # Decouple arguments and validate
    # TODO: Verify if assertion errors works as expected.
    cell_label, heading = pose
    assert cell_label in world, 'opertion fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'opertion fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {SOUTH: (0, -1), WEST: (1, 0), NORTH: (0, 1), EAST: (-1, 0)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), (heading - 90) % 360

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return None


def left(world, pose):
    """
    Returns new pose after performing right operation on pose.
    Definition of right is - moves to right cell and turns by -90 deg.

    :param world: simworld.World object
    :param pose: 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    :return: pose 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    """
    # Decouple arguments and validate
    # TODO: Verify if assertion errors works as expected.
    cell_label, heading = pose
    assert cell_label in world, 'operation fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'operation fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {SOUTH: (0, 1), WEST: (-1, 0), NORTH: (0, -1), EAST: (1, 0)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), (heading + 90) % 360

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return None


def fwd_right(world, pose):
    """
    Returns new pose after performing right operation on pose.
    Definition of right is - moves to right cell and turns by -90 deg.

    :param world: simworld.World object
    :param pose: 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    :return: pose 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    """
    # Decouple arguments and validate
    # TODO: Verify if assertion errors works as expected.
    cell_label, heading = pose
    assert cell_label in world, 'opertion fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'opertion fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {WEST: (1, -1), NORTH: (1, 1), EAST: (-1, 1), SOUTH: (-1, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return None


def fwd_left(world, pose):
    """
    Returns new pose after performing right operation on pose.
    Definition of right is - moves to right cell and turns by -90 deg.

    :param world: simworld.World object
    :param pose: 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    :return: pose 2-tuple of (cell_label <integer>, heading direction <MACRO:integer multiple of 90 in [0, 360)>)
    """
    # Decouple arguments and validate
    # TODO: Verify if assertion errors works as expected.
    cell_label, heading = pose
    assert cell_label in world, 'opertion fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'opertion fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {EAST: (1, 1), SOUTH: (-1, 1), WEST: (-1, -1), NORTH: (1, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return None


def wait(world, pose):
    return pose


# Define action set for cars.
CAR_ACTIONS = {fwd: 10, right: 30, left: 40, fwd_right: 20, fwd_left: 20, wait: 50}


class Car(sm.SM):
    def __init__(self):
        pass

    def getNextValues(self, state, inp):
        pass


class StopSign(sm.SM):
    """
    Defines Stop Sign behavior as follows.
        1. If car arrives at new stop sign - wait for at least 1 time step.
        2. Wait until
            a. All previously waiting cars at intersection have made a move (any move)
        OR  b. Car has already waited for N-time steps

    Definition of SM:
        1. State: 2-tuple of (last seen location, waitTime)
        2. Input: pose
        3. Output: Boolean - True if stopping is requested, False otherwise.
    """
    def __init__(self, maxWait=5):
        """
        Instantiates Stop-Sign Behavior.

        :param maxWait: Non-negative Integer - Maximum waiting time for car, after which
         it may switch to aggressively cross the intersection.
        """
        # Internal variables
        self.maxWait = maxWait

        # Initialize state
        self.currState = None

    def getNextValues(self, state, inp):
        """
        Transition function definition.

        :param state: 2-tuple (last pose of car, last seen slice of world, label of last seen slice, waitTime)
        :param inp: 3-tuple of (car pose, world slice <2D np.array>, labels of slice <dict of {label: CellAP object}>)
        :return: Boolean -> True,  if stop action is recommended. False, otherwise.

        @remark: pose = (label of cell <Integer>, angle of heading <degrees in [0, 360)>)
        """
        # Decouple arguments
        (newLoc, newHeading), worldSlice, labelSlice = inp
        (oldLoc, oldHeading), oldSlice, oldLabels, waitTime = state

        # Check if current location has a stop sign
        isStopSign = labelSlice[newLoc].stopSign

        # If there is no stop sign, ignore
        if not isStopSign:
            nState = (inp, 0)
            nOut = False

        # If we are at stop sign then check if we were here in previous step
        elif newLoc == oldLoc and waitTime < self.maxWait:
            nState = (inp, waitTime + 1)
            nOut = True

        else:   # we have waited for enough time, let's aggressively cross the stop sign
            nState = (inp, waitTime + 1)
            nOut = False

        return nState, nOut


class Go2Goal(sm.SM):
    def __init__(self, world, actions):
        """
        Constructs Go2Goal Behavior machine.
        :param world: simworld.World object
        :param actions: dictionary of {action_fcn: weight}
        """
        #  Preprocess (Normalize action weights)
        maxWt = max([actions[act] for act in actions.keys()])
        for act in actions.keys():
            actions[act] /= maxWt

        # Local variables
        self.world = world
        self.actions = actions

        # Initialize state
        self.startState = None
        self.initialize()

    def getNextValues(self, state, inp):
        """
        Transition function.

        :param state: None (Memory less)
        :param inp: 2-tuple of (suggestedMove, pose)
        :return: 2-tuple of (None, 2-tuple of (selected action, output of action))
        """
        # Decouple arguments
        suggestedMove, pose = inp

        # Compute reachable set
        reachableSet = self._computeReachableSet(pose)

        # If suggestedMove is reachable, take it
        for rchLoc in reachableSet:
            action, (pos, head) = rchLoc
            if suggestedMove == pos:
                return None, (action, (pos, head))

        # Else, suggested move is not reachable: select least cost feasible action
        feasibleActions = [self.actions[act] for act, _ in reachableSet]
        cheapestActIdx = feasibleActions.index(min(feasibleActions))
        action, (pos, head) = reachableSet[cheapestActIdx]
        return None, (action, (pos, head))

    def _computeReachableSet(self, pose):
        """
        Computes the reachable set from given pose assuming world, action set is already known.

        :param pose: (label of cell <integer>, heading angle <range [0, 360)>)
        :return: list of 2-tuple of (action, poses) reachable.
        """
        reachableSet = list()
        for act in self.actions.keys():
            nPose = act(self.world, pose)
            if nPose is None:
                continue

            reachableSet.append((act, nPose))

        return reachableSet


class AvoidObstacle(sm.SM):
    pass


class Router(sm.SM):
    REROUTED = 'rerouted'
    ON_PATH = 'on path'

    def __init__(self, world, aggression):
        """
        Constructs a Router state machine.
        :param world: simworld.World object
        :param aggression: dict of cost structure:
            {'r2r': float,  # road2road
             'r2g': float,  # road2grass
             'r2o': float,  # road2obstacle
             'g2r': float,  # grass2road
             'g2g': float,  # grass2grass
             'g2o': float,  # grass2obstacle
            }

        These costs define the aggression of car. For example,
        if cost of road2grass = 20 and road2road = 10, then car will
        prefer to go over grass if length of road is 2 cells more than
        that of driving on grass.

        In essence, the difference between the costs is as important as
        the costs themselves.
        """
        # Internal variables
        self.grf = self._graphify(world, aggression)

        # Initialize state
        self.startState = (None, list(), None)        # State = 3-tuple (goal, route, None/REROUTED/ON_PATH)
        self.initialize()

    def updateAggression(self, world, aggValue):
        """
        The function may be introduced later to update weights of
        internal graph.

        :param aggValue: dictionary as mentioned in description of constructor.
        :return: None
        """
        raise NotImplementedError('Router.updateAggression: Is not implemented in this version of project.')

    def getNextValues(self, state, inp):
        """
        Transition Function.

        :param state: 2-tuple (goal <integer>, last computed path <list>)
        :param inp: 2-tuple (goal <integer>, car pose)
        :return: suggested next cell label <integer>
        """
        # Decouple arguments
        goal, route, lastState = state
        nGoal, (pos, heading) = inp

        # No prior path available
        if lastState is None:
            nRoute = nx.astar_path(G=self.grf, source=pos, target=nGoal)
            nState = Router.REROUTED

        # Else, if last computed route is known
        else:
            # Reroute if goal has changed or we have deviated route
            if nGoal != goal or pos not in route:
                nRoute = nx.astar_path(G=self.grf, source=pos, target=nGoal)
                nState = Router.REROUTED

            # Else make no change
            else:
                nState = Router.ON_PATH
                nRoute = route

        if len(nRoute) > 1:
            nRoute.pop(0)

        return (nGoal, nRoute, nState), nRoute[0]

    def _graphify(self, world, aggression):
        """
        Generates graph representation of the world, and defines
        cost of edges based on aggression.

        :param world: simworld.World object.
        :param aggression: dictionary as described in constructor of Router class.
        :return:
        """
        assert isinstance(world, simworld.World), 'Router._graphify: world must be simworld.World object.'

        # Initialize graph
        grf = nx.DiGraph()

        # Get dimensions of world
        dim = world.dim

        # For all cells in the world do
        for i in range(dim[0]):
            for j in range(dim[1]):
                # Get the valid neighbors of the cell (Can be made efficient later!)
                neighbors = list()
                if (i, j) in world: neighbors.append(world.label((i, j)))
                if (i, j+1) in world: neighbors.append(world.label((i, j+1)))
                if (i, j-1) in world: neighbors.append(world.label((i, j-1)))
                if (i+1, j) in world: neighbors.append(world.label((i+1, j)))
                if (i+1, j-1) in world: neighbors.append(world.label((i+1, j-1)))
                if (i+1, j+1) in world: neighbors.append(world.label((i+1, j+1)))
                if (i-1, j) in world: neighbors.append(world.label((i-1, j)))
                if (i-1, j-1) in world: neighbors.append(world.label((i-1, j-1)))
                if (i-1, j+1) in world: neighbors.append(world.label((i-1, j+1)))


                # For each neighbor, find the cost of edge
                fromLabel = world.label((i, j))
                for toLabel in neighbors:
                    fromAP = world.ap(fromLabel)
                    toAP = world.ap(toLabel)

                    # Enumerate Exhaustively to check which case we have
                    if fromAP.isRoad and toAP.isRoad:
                        cost = aggression['r2r']

                    elif fromAP.isRoad and not toAP.isRoad and not toAP.isStatObs:
                        cost = aggression['r2g']

                    elif fromAP.isRoad and toAP[simworld.STAT_OBS]:
                        cost = aggression['r2o']

                    elif not fromAP.isRoad and not fromAP.isStatObs and toAP.isRoad:
                        cost = aggression['g2r']

                    elif not fromAP.isRoad and not fromAP.isStatObs and not toAP.isRoad and \
                            not toAP.isStatObs:
                        cost = aggression['g2g']

                    elif not fromAP.isRoad and not fromAP.isStatObs and toAP.isStatObs:
                        cost = aggression['g2o']

                    else:
                        print('Router.graphify: Something weird is happening!')
                        cost = float('Inf')

                    # Add nodes
                    grf.add_edge(fromLabel, toLabel, weight=cost)

        return grf


if __name__ == '__main__':

    labels = {'isRoad': np.ones((3, 3)), 'isStatObs': np.ones((3, 3)), 'stopSign': np.zeros((3, 3))}
    myWorld = simworld.World(labels)

    agg = {
        'r2r': 50,
        'r2g': 100,
        'r2o': float('Inf'),
        'g2r': 0,
        'g2g': 100,
        'g2o': float('Inf'),
    }
    r = Router(myWorld, agg)

    gotogoal = Go2Goal(myWorld, actions=CAR_ACTIONS)

    print(fwd(myWorld, (1, NORTH)))
    print(myWorld.labelMap())
    print(gotogoal.step((8, (1, NORTH))))

