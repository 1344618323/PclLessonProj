import math


class DistIndex:
    def __init__(self, dist, index):
        self.dist = dist
        self.index = index


class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.dist_index = []
        self.worst_dist = math.inf
        for i in range(capacity):
            self.dist_index.append(DistIndex(math.inf, -1))

    def insert_node(self, dist, index):
        if dist >= self.worst_dist:
            return
        if self.count < self.capacity:
            self.count += 1
        i = self.count - 1
        while i >= 1:
            if dist < self.dist_index[i - 1].dist:
                self.dist_index[i] = self.dist_index[i - 1]
                i -= 1
            else:
                break
        self.dist_index[i] = DistIndex(dist, index)
        self.worst_dist = self.dist_index[self.capacity - 1].dist

    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index):
            output += '{} - {}\n'.format(dist_index.index, dist_index.dist)
        return output


class RNNResultSet:
    def __init__(self, radius):
        self.dist_index = []
        self.worst_dist = radius

    def insert_node(self, dist, index):
        if dist > self.worst_dist:
            return
        self.dist_index.append(DistIndex(dist, index))

    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index):
            output += '{} - {}\n'.format(dist_index.index, dist_index.dist)
        return output
