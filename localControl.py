import numpy as np
import car


if __name__ == "__main__":
	waitCounter = 0;
	stopSignCounter = 0;
	Car car;
	while car.goal != car.position
		if isDanger(localMap, car.position)
			alertDriver('danger %f', dangerCounter)
			car.immediateAction = crossProduct(localMap, car.position)
			if car.immediateAction is null
				alertDriver('death')
				return
			execute(car.immediateAction)
			if car.immediateAction == car.actions.wait
				waitCounter = waitCounter +1
				if waitCounter > waitThreshhold
					car.replan()
					waitCounter = 0
		else if car.position == car.nextGoal
			waitCounter = 0
			if redYellowLight()
				execute(car.actions.wait)
			else if stopSign(stopSignCounter)
				execute(car.actions.wait)
				stopSignCounter = stopSignCounter+1
			else
				stopSignCounter=0
				if isDanger(localMap, car.nextAction)
					execute(car.actions.wait)
					waitCounter = waitCounter +1
					if waitCounter > waitThreshhold
						car.replan()
						waitCounter = 0
				else
					execute(car.nextAction)
					car.arrived()
		else
			waitCounter = 0
			execute(car.actions.follow)
