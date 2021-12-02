# pylint: disable=missing-module-docstring

import numpy as np
from result_set import KNNResultSet, RNNResultSet


class Node:
    def __init__(self, key, value):
        self.left = None
        self.right = None
        self.key = key
        self.value = value


def insert(root: Node, key, value=-1):
    if root is None:
        root = Node(key, value)
    elif root.key > key:
        root.left = insert(root.left, key, value)
    elif root.key < key:
        root.right = insert(root.right, key, value)
    return root


def knn_search(root: Node, query, result: KNNResultSet):
    if root is None:
        return False
    result.insert_node(abs(query - root.key), root.value)
    if result.worst_dist == 0:
        return True
    if query >= root.key:
        if knn_search(root.right, query, result):
            return True
        elif result.worst_dist > abs(query - root.key):
            return knn_search(root.left, query, result)
    else:
        if knn_search(root.left, query, result):
            return True
        elif result.worst_dist > abs(query - root.key):
            return knn_search(root.right, query, result)
    return False


def rnn_search(root: Node, query, result: RNNResultSet):
    if root is None:
        return False
    result.insert_node(abs(query - root.key), root.value)
    if result.worst_dist == 0:
        return True
    if query >= root.key:
        if rnn_search(root.right, query, result):
            return True
        elif result.worst_dist > abs(query - root.key):
            return rnn_search(root.left, query, result)
    else:
        if rnn_search(root.left, query, result):
            return True
        elif result.worst_dist > abs(query - root.key):
            return rnn_search(root.right, query, result)
    return False


if __name__ == '__main__':
    db_size = 100
    db = np.random.permutation(db_size).tolist()
    root = None
    for i in range(len(db)):
        root = insert(root, db[i], i)
    knn_result = KNNResultSet(6)
    knn_search(root, 0, knn_result)
    rnn_result = RNNResultSet(6)
    rnn_search(root, 5, rnn_result)
    for item in rnn_result.dist_index:
        print(item.index, db[item.index])
