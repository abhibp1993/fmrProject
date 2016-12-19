#!/usr/bin/env python
"""
Author: Abhishek Kulkarni, Ari Goodman, Yicong Xu
Modified: December 16, 2016

Defines car with discrete dynamics for simulation.
"""

# Imports
import sm


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
    increment_row_col = {EAST: (1, 0), NORTH: (0, 1), WEST: (-1, 0), SOUTH: (0, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return pose


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
    increment_row_col = {EAST: (0, -1), NORTH: (1, 0), WEST: (0, 1), SOUTH: (-1, 0)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), (heading - 90) % 360

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return pose


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
    assert cell_label in world, 'opertion fwd: received pose{0} outside of world.' % pose
    assert heading % 90 == 0 and 0 <= heading < 360, 'opertion fwd: Received unacceptable angle={0}.' % heading

    # Define operation {current_heading: (add2row, add2col)}
    increment_row_col = {EAST: (0, 1), NORTH: (-1, 0), WEST: (0, -1), SOUTH: (1, 0)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), (heading + 90) % 360

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return pose


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
    increment_row_col = {EAST: (1, -1), NORTH: (1, 1), WEST: (-1, 1), SOUTH: (-1, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return pose


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
    increment_row_col = {EAST: (1, 1), NORTH: (-1, 1), WEST: (-1, -1), SOUTH: (1, -1)}

    # Operate
    cur_cell = world.cell(cell_label)
    new_cell = (cur_cell[0] + increment_row_col[heading][0], cur_cell[1] + increment_row_col[heading][1])

    if new_cell in world:
        return world.label(new_cell), heading

    print('operation fwd:: Warning: Operation leading to outside world. Canceling operation.')
    return pose


def wait(world, pose):
    return pose


# Define action set for cars.
CAR_ACTIONS = {fwd: 1, right: 2, left: 3, fwd_right: 1, fwd_left: 1}


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
    pass


class AvoidObstacle(sm.SM):
    pass


class Router(sm.SM):
    def __init__(self):
        pass


