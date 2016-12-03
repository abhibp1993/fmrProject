__author__ = 'Andy Xu, Ari Goodman'


class LabeledEdges:
    def __init__(self, data=[False, False, False, False, False, True]):
        self.yieldSign = data[0]
        self.stopSign = data[1]
        self.illegal = data[2]
        self.obstacle = data[3]
        self.road = data[4]
        self.grass = data[5]
        self.values = [self.yieldSign, self.stopSign, self.ill  egal, self.obstacle, self.road, self.grass]