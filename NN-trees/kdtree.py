import numpy as np
import math
from result_set import KNNResultSet, RNNResultSet
import scipy.spatial.kdtree
import time


class Node:
    def __init__(self, axis, value, left, right, point_indices: np.ndarray):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        return False

    def __str__(self):
        output = 'axis:{} ,value:{}, point_indices:{} '.format(self.axis, self.value, self.point_indices.tolist())
        return output


def kdtree_recursive_construct(root: Node, db: np.ndarray, axis, point_indices: np.ndarray, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)
    if point_indices.shape[0] > leaf_size:
        sorted_result = np.argsort(db[point_indices, axis])
        sorted_point_indices = point_indices[sorted_result]
        middle_left_idx = math.ceil(sorted_result.shape[0] * 0.5) - 1
        middle_right_idx = math.ceil(sorted_result.shape[0] * 0.5)
        middle_val = (db[sorted_point_indices[middle_left_idx], axis] + db[
            sorted_point_indices[middle_right_idx], axis]) * 0.5
        root.value = middle_val

        if axis == db.shape[1] - 1:
            axis = 0
        else:
            axis += 1
        root.left = kdtree_recursive_construct(root.left, db, axis, sorted_point_indices[0:middle_right_idx], leaf_size)
        root.right = kdtree_recursive_construct(root.right, db, axis, sorted_point_indices[middle_right_idx:],
                                                leaf_size)
    return root


def kdtree_construct(db: np.ndarray, leaf_size):
    root = None
    root = kdtree_recursive_construct(root, db, 0, np.arange(db.shape[0]), leaf_size)
    return root


def kdtree_traverse(root: Node):
    if root is None:
        print('None')
    elif root.is_leaf():
        print(root)
    else:
        print(root)
        kdtree_traverse(root.left)
        kdtree_traverse(root.right)


def kdtree_knn_search(root: Node, db: np.ndarray, query: np.ndarray, result_set: KNNResultSet):
    if root is None:
        return
    elif root.is_leaf():
        for idx in root.point_indices:
            point = db[idx]
            result_set.insert_node(np.linalg.norm(point - query), idx)
    else:
        if query[root.axis] > root.value:
            kdtree_knn_search(root.right, db, query, result_set)
            if result_set.worst_dist > (query[root.axis] - root.value):
                kdtree_knn_search(root.left, db, query, result_set)
        else:
            kdtree_knn_search(root.left, db, query, result_set)
            if result_set.worst_dist > (root.value - query[root.axis]):
                kdtree_knn_search(root.right, db, query, result_set)


def kdtree_rnn_search(root: Node, db: np.ndarray, query: np.ndarray, result_set: RNNResultSet):
    if root is None:
        return
    elif root.is_leaf():
        for idx in root.point_indices:
            point = db[idx]
            result_set.insert_node(np.linalg.norm(point - query), idx)
    else:
        if query[root.axis] > root.value:
            kdtree_rnn_search(root.right, db, query, result_set)
            if result_set.worst_dist > (query[root.axis] - root.value):
                kdtree_rnn_search(root.left, db, query, result_set)
        else:
            kdtree_rnn_search(root.left, db, query, result_set)
            if result_set.worst_dist > (root.value - query[root.axis]):
                kdtree_rnn_search(root.right, db, query, result_set)


if __name__ == '__main__':
    db_size = 64
    dim = 3
    db = np.random.rand(db_size, dim)
    leaf_size = 4
    root = kdtree_construct(db, leaf_size)
    kdtree_traverse(root)

    k = 5
    result_set = KNNResultSet(capacity=k)
    query = np.array([0, 0, 0])
    kdtree_knn_search(root, db, query, result_set)
    print('Our impl result:')
    print(result_set)

    diff = np.linalg.norm(np.expand_dims(query, axis=0) - db, axis=1)
    sort_result = np.argsort(diff)
    print('Brute force result:')
    print(sort_result[0:k])
    print(diff[sort_result[0:k]])

    sci_kdtree = scipy.spatial.KDTree(db, leafsize=leaf_size)
    dists, indices = sci_kdtree.query(query, k=k)
    # Experiments show that the implementation of scipy is much faster
    print('Scipy KDtree:')
    print(dists)
    print(indices)

    result_set = RNNResultSet(radius=0.5)
    query = np.array([0, 0, 0])
    kdtree_rnn_search(root, db, query, result_set)
    print('Our impl result:')
    print(result_set)

    indices = sci_kdtree.query_ball_point(query, r=0.5)
    print('Scipy KDtree:')
    print(indices)

    # db_size = 64000
    # dim = 3
    # db = np.random.rand(db_size, dim)
    # leaf_size = 4
    # root = kdtree_construct(db, leaf_size)
    # sci_kdtree = scipy.spatial.KDTree(db, leafsize=4)

    # k = 5
    # start_t = time.time()
    # for i in range(10000):
    #     result_set = KNNResultSet(capacity=k)
    #     query = np.random.rand(dim)
    #     kdtree_knn_search(root, db, query, result_set)
    # print('Our impl: {}'.format(time.time() - start_t))
    # start_t = time.time()
    # for i in range(10000):
    #     query = np.random.rand(dim)
    #     sci_kdtree.query(query, k=k)
    # print('Scipy impl: {}'.format(time.time() - start_t))

    # start_t = time.time()
    # for i in range(50):
    #     result_set = RNNResultSet(radius=0.5)
    #     query = np.random.rand(dim)
    #     kdtree_rnn_search(root, db, query, result_set)
    # print('Our impl: {}'.format(time.time() - start_t))
    # start_t = time.time()
    # for i in range(50):
    #     query = np.random.rand(dim)
    #     sci_kdtree.query_ball_point(query, r=0.5)
    # print('Scipy impl: {}'.format(time.time() - start_t))
