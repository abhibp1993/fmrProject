import graphds
import sm

NORTH = 'north'
SOUTH = 'south'
EAST  = 'east'
WEST  = 'west'


# Possible Actions:
def stop(state):
    return state

def goStraight(state):
    pass

def turnLeft(state):
    pass

def turnRight(state):
    pass

def uTurn(state):
    pass

def replan(state):
    pass


# Class Car
class Car(sm.SM):
    def __init__(self, initPose, speed, goalLoc, graphWorld, behavior):
        # State of Car: [x, y, dir, speed]
        self.startState = initPose.append(speed)

        # World and Routes
        self.goal = goalLoc
        self.world = graphWorld

        # Set the behavior for car and initialize
        self.behavior = behavior
        self.behavior.route = None
        self.behavior.initialize()

    def generatePlan(self):
        """
        Top Level Graph (A*) Route Planner.
        :return:
        """
        return list()

    def act(self, action, state):
        """
        Applies the action to state to generate a new state.
        Note: No validitiy checking of action is done. So output state may be out of bounds of map.

        :param action: function pointer.
        :param state: [x, y, dir, speed]
        :return: new state [x, y, dir, speed]
        """
        return action(state)

    def seeWorld(self, grid, location):
        """
        returns dictionary with each obserable cell along with corresponding Atomic propositions

        @param grid: Grid Class
        @param location: valid cell in Grid.
        @return: Dictionary of key=cell, value=label of that cell
        """
        keys = [[i, j] for i in range(-2, 3) for j in range(-2, 3)]
        vision = dict()
        vision.keys().append(keys)

        
        return vision

    def getNextValues(self, state, inp):

        # Make easy reference for state variables
        x = state[0]
        y = state[1]

        # If we don't have a path, Find one.
        if self.route is None:
            self.route = self.generatePlan()

        # Find visible world, and update local APs
        visibleWorld = self.seeWorld(self.world.grid, [x, y])

        # update behavior machine
        action = self.behavior.step([state, visibleWorld])

        # Execute action
        nState = self.act(action, state)    # returns [x, y, dir, speed]

        return [nState, None]


class SDC(sm.SM):
    MOVE = 'move'
    STOP_SIGN = 'stop_sign'
    WAIT = 'wait'

    MAX_WAIT_TIME = 5

    def __init__(self):
        """
        Instantiates the Self-driving car behavior.
        """
        # Start state of car (behavior)
        self.startState = [SDC.WAIT, None, 0]   # state = [Mode, Edge (of route), waitTime]

        # Route of Car = Desired top-level behavior
        self.route = None

    def getNextValues(self, state, inp):
        """
        Transition function for Car State Machine.

        @param state: [x, y, speed, direction]
        @param inp: list of cells from world that are observable.
        @return: (nState, action)
        """
        mode = state[0]
        edge = state[1]
        waitTime = state[2]

        # Wait Behavior
        if self.route is None:
            return [[mode, edge, 0], stop]

        if mode == SDC.WAIT:
            action = self.action(edge, inp[1])  # inp[1] = grid world
            if action is not None:
                mode = SDC.MOVE
                waitTime = 0
                return [[mode, edge, waitTime], action]
            else:
                return [[state, edge, waitTime+1], stop]

        # Move Behavior
        if mode == SDC.MOVE and not self.nextCellStopSign(inp):
            action = self.action(edge, inp)
            if action is not None:
                return [[mode, edge, 0], action]
            else:
                mode = SDC.WAIT
                return [[mode, edge, 0], stop]

        elif mode == SDC.MOVE: # and stop sign is in next cell
            mode = SDC.STOP_SIGN
            return [[mode, edge, 0], stop]

        # Stop-Sign Behavior
        if mode == SDC.STOP_SIGN and self.isIntersectionClear(inp):
            action = self.action(edge, inp)
            if action is not None:
                return [[mode, edge, 0], action]
            else:
                mode = SDC.WAIT
                return [[mode, edge, 0], stop]

        elif mode == SDC.STOP_SIGN:
            return [[mode, edge, 0], stop]

    def action(self, edge, inp):
        """
        Computes desired action based on path-plan and observable-world.

        :param inp: list of observable cells of world
        :return: function pointer to suitable action.
        """
        return goStraight

    def nextCellStopSign(self, inp):
        """
        Checks if next cell contains a stop-sign or not.

        :param inp: list of observable cells.
        :return: Boolean
        """
        return False

    def isIntersectionClear(self, inp):
        """
        Checks if the adjacent cells to intersection that are road are not-occupied.

        :param inp: list of observable cells
        :return: Boolean
        """
        return True


'''
# class Car1(sm.SM):
#
#     actions = ['follow', 'changeLeft', 'changeRight', 'turnRight', 'turnLeft', 'uTurn', 'wait']
#     visual_radius = 5
#
#     def __init__(self, initPos, goalPos, globalMap):
#         self.position = initPos     # change to state
#         self.goal = goalPos
#         self.globalMap = globalMap
#         self.localMap = self.vision(globalMap)
#         self.plan = self.planPath(goalPos, globalMap)
#         self.immediateAction = self.actions.wait
#         self.action = self.plan[0]
#
#     def vision(self, globalMap):
#         return globalMap.slice([self.state.x, self.state.y, self.visual_radius, self.visual_radius])
#
#     def planPath(self, goalPos, globalMap):
#         path = graphds.aStar( self.graphMap, [self.state.x, self.state.y], self.goal )
#         edges = graphds.edgify(self.graphMap, path)
#
#         return edges
#
#     def getNextValues(self, state, sensorInput):
#
#         if [state.x, state.y] == self.goal:
#             return [[state.x, state.y], 'wait']  #convert wait to function
#
#         # Check danger
#         if self.danger(self.localMap, sensorInput):
#             print 'AlERT: DANGER!'
#             action = self.findAction(sensorInput)
#             nState = self.transition(action)
#
#             if len(nState) == 0:
#                 print 'NULL TRANSITION..'
#
#             if state.speed == 0 and action == 'wait':
#                 self.waitCounter += 1
#                 if self.waitCounter > self.maxWaitTime:
#                     self.replan()
#                     self.waitCounter = 0
#
#         # Check intersections
#         # TODO: The stop-sign behavior needs to go in findAction
#         if state.currentEdge.dest.cell == [state.x, state.y]:
#             edge = self.edgeTransfer(self.plan, state.currentEdge)
#             if edge.isInter and edge.ap[graphds.Edge.ACTIVE]:
#                 action = self.findAction(sensorInput)
#                 nState = None           # Construct new state
#
#
#
#
#
#
#
#
#     def edgeTransfer(self, plan, currEdge):
#         """
#         Finds a next edge once we have traveled current edge
#
#         @optimization: Path as queue => popping is easy!
#         """
#         idx = plan.index(currEdge)
#         return plan[idx+1]
'''