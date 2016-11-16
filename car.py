class Car:
	actions = {follow, changeLeft, changeRight, turnRight, turnLeft, uTurn, wait}
	visual_radius = 5
	def __init__(self, initPos, goalPos, globalMap):
		self.position = initPos
		self.goal = goalPos
		self.globalMap = globalMap
		self.localMap = self.vision(globalMap)
		self.plan = self.planPath(goalPos, globalMap)
		self.immediateAction = self.actions.wait
		self.action = self.plan[1]
	def vision(self, globalMap)
		print 'TODO'
	def planPath(self, goalPos, globalMap)
		print 'TODO'
	
