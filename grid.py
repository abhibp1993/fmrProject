"""
Author: Abhishek Kulkarni
Modified: November 14, 2016

Implements basic data structures world creation
"""


class Grid(object):
    GRASS = 'grass'
    ROAD = 'road'
    STOP = 'stop'
    OCC = 'occupied'
    NEXTOCC = 'nextOcc'     # possible locations which may be occupied in next time step.

    def __init__(self, dim=20):
        """
        Creates a dim x dim sized box world with default grass everywhere.

        @param dim: size of the discrete world.
        """
        self.dim = dim
        self.AP = {Grid.GRASS: list(), Grid.ROAD: list(), Grid.STOP: list(), Grid.OCC: list(), Grid.NEXTOCC: list()}

        for i in range(dim):
            for j in range(dim):
                self.AP[Grid.GRASS].append([i, j])

    def addRoad(self, cells):
        """
         Sets the given cells as road. The road doesn't need to be straight.
         Any turns, will be taken care of in planning algorithm.
        """
        for c in cells:
            try:
                self.AP[Grid.GRASS].remove(c)
            except:
                pass

        for c in cells:
            try: self.AP[Grid.ROAD].remove(c)
            except: pass
            finally: self.AP[Grid.ROAD].append(c)

    def addStop(self, cell):
        """
        Adds a stop sign to cell.
        Stop sign can only be added if the cell is a road.

        @param cell: valid (i, j)
        """
        if cell in self.AP[Grid.ROAD]:
            self.AP[Grid.STOP].append(cell)

    def slice(self, rect):
        pass

    def __str__(self):
        return str(self.AP)


# Basic Testing
if __name__ == '__main__':
    g = Grid(4)
    print g

    g.addRoad([[1, i] for i in range(3)])
    g.addStop([1, 0])
    g.addStop([3, 1])

    print g