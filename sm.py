class SM(object):
    """
    Represents the basic StateMachine. The class provides a base for implementation of
    all state machines. This class MUST NOT be instantiated but classes may be derived
    out of this class.

    Global Variables:
        startState: a generic start state of stateMachine, default: None

    """
    startState = None
    doneCondition = None

    def initialize(self):
        """
        Initializes the machine with currState variable to startState.
        [Optionally, when debugger is introduced, it must reset the logging related lists.]
        """
        self.currState = self.startState

    def getNextValues(self, state, inp):
        """
        Returns the output which should result IF the given input is applied to the machine
        IF it were in given state. In other words, returns (but not performs) the transition
        on state machine.

        @state: valid state of machine
        @inp: input provided to machine

        @return: 2-tuple of (newState, output)
        [For more complete description, refer documentation]
        """
        raise NotImplementedError("getNextValues method is not implemented")

    def step(self, inp):
        """
        Advances the machine my one step. Applies the argument - inp - to current state.
        The resultant state is the new state of machine.
        [Internal function. Do not make use of it/tweak it.]

        @inp: input to machine
        @return: the output of the transition performed on the machine.

        Errors:
            1. ValueError: Too many values to unpack - If getNextValues does not return a 2-tuple
            2. ValueError: Too less values to unpack - If getNextValues does not return a 2-tuple

        """
        nState, output = self.getNextValues(self.currState, inp)
        self.currState = nState
        return output

    def transduce(self, inpSequence=list()):
        """
        Advances machine by applying inpSequence as input to machine.
        Index - 0 being the first applied input.

        @inpSequence: ordered list of n-inputs
        @returns: ordered list of n-outputs resulting from n-transitions

        Errors:
            1. ValueError: Empty Input Sequence

        """

        lstOutputs = list()

        # Initialize the state machine
        self.initialize()

        # Check if inpSequence is non-empty
        if len(inpSequence) == 0:
            raise ValueError("Empty Input Sequence")

        # Loop and apply the inputs
        for i in inpSequence:
            lstOutputs.append(self.step(i))
        # try:
        #                lstOutputs.append(self.step(i))
        #            except Exception, ex:
        #                lstOutputs.append(None)
        #                print 'Step function failed, for input: ', i, ' ', ex.message

        return lstOutputs

    def transduceFcn(self, inpFcn, nSteps=[]):
        """
        Advances machine by calling input function with next-index of loop.

        @inpFcn: a procedure with 1 positive-integer parameter
        @nSteps: list of +ve-integer indices which should be the inputs applied. [Eg. nSteps = range(1, 10)]
        @returns: ordered list of n-outputs resulting from n-transitions

        Errors:
            1. ValueError: Empty Input Sequence

        """
        lstOutputs = list()
        # Loop and apply the inputs
        for i in nSteps:
            try:
                lstOutputs.append(self.step(inpFcn(i)))
            except:
                lstOutputs.append(None)
                print('Step function failed, for index: ', i)

        return lstOutputs

    def done(self):
        """
        Evaluates if the machine is "done" it's working -- i.e. Is the Target Reached?

        @returns: True if doneCondition evaluates to true.
        [ doneCondition needs to be initialized using sm.initialize(condition) ]

        """
        try:
            if self.doneCondition(self.currState): return True
        except:
            # Add to log
            print("WARNING: doneCondition looks to have error OR is not initialized")

        return False

    def run(self):
        ### figure out what and how to choose inputs??

        # nState = self.startState
        # while not self.done(self.currState):
        #    output = self.step(???inp???)
        #    print output

        raise NotImplementedError

    def __str__(self):
        try:
            return 'State Machine: Current State', str(self.currState)
        except:
            return 'State Machine: Not yet initialized'

