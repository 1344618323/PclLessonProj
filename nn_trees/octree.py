import numpy as np
from .result_set import KNNResultSet, RNNResultSet
import time


class Octant:
    def __init__(self, center, extent, point_indices):
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = False
        self.children = [None for i in range(8)]

    def __str__(self):
        output = ''
        output += 'center: {}, '.format(self.center)
        output += 'extent: {}, '.format(self.extent)
        output += 'is_leaf: {}, '.format(self.is_leaf)
        output += 'indices_size: {}'.format(len(self.point_indices))
        return output


def octree_recursive_construction(root: Octant, db: np.ndarray, center, extent, point_indices, leaf_size, min_extent):
    if root is None:
        root = Octant(center, extent, point_indices)

    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
        return root

    c_point_indices = [[] for i in range(8)]
    for idx in point_indices:
        point = db[idx]
        morton_code = 0
        if point[0] > center[0]:
            morton_code |= 1
        if point[1] > center[1]:
            morton_code |= 2
        if point[2] > center[2]:
            morton_code |= 4
        c_point_indices[morton_code].append(idx)
    factor = [-0.5, 0.5]
    for i in range(8):
        indices = c_point_indices[i]
        if len(indices) == 0:
            continue
        c_center = np.zeros_like(center)
        c_center[0] = factor[(i & 1) > 0] * extent + center[0]
        c_center[1] = factor[(i & 2) > 0] * extent + center[1]
        c_center[2] = factor[(i & 4) > 0] * extent + center[2]
        root.children[i] = octree_recursive_construction(root.children[i], db, c_center, 0.5 * extent, indices,
                                                         leaf_size, min_extent)
    return root


def octree_construction(db: np.ndarray, leaf_size, min_extent):
    min_val = np.min(db, axis=0)
    max_val = np.max(db, axis=0)
    extent = np.max(max_val - min_val) * 0.5
    center = extent + min_val
    root = None
    root = octree_recursive_construction(root, db, center, extent, range(db.shape[0]), leaf_size, min_extent)
    return root


def traverse_octree(root: Octant):
    if root is None:
        print('None')
    elif root.is_leaf:
        print(root)
    else:
        print(root)
        for child in root.children:
            traverse_octree(child)


def inside(query: np.ndarray, radius: float, octant: Octant):
    offset = np.fabs(octant.center - query)
    return np.all((offset + radius) < octant.extent)


def overlap(query: np.ndarray, radius: float, octant: Octant):
    offset = np.fabs(octant.center - query)
    if np.any(offset > (octant.extent + radius)):
        return False
    if np.sum(offset < octant.extent).astype(int) >= 2:
        return True
    diff_x = max(offset[0] - octant.extent, 0)
    diff_y = max(offset[1] - octant.extent, 0)
    diff_z = max(offset[2] - octant.extent, 0)
    return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z < radius * radius


def contain(query: np.ndarray, radius: float, octant: Octant):
    offset = np.fabs(octant.center - query)
    return np.linalg.norm(offset + octant.extent) < radius


def octree_knn_search(root: Octant, db: np.ndarray, query: np.ndarray, reasult_set: KNNResultSet):
    if root is None:
        return False

    elif root.is_leaf:
        assert len(root.point_indices) > 0
        leaf_points = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            reasult_set.insert_node(diff[i], root.point_indices[i])
        return inside(query, reasult_set.worst_dist, root)

    morton_code = 0
    if query[0] > root.center[0]:
        morton_code |= 1
    if query[1] > root.center[1]:
        morton_code |= 2
    if query[2] > root.center[2]:
        morton_code |= 4

    if octree_knn_search(root.children[morton_code], db, query, reasult_set):
        return True
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        elif not overlap(query, reasult_set.worst_dist, child):
            continue
        elif octree_knn_search(child, db, query, reasult_set):
            return True
    return inside(query, reasult_set.worst_dist, root)


def octree_rnn_search(root: Octant, db: np.ndarray, query: np.ndarray, reasult_set: RNNResultSet):
    if root is None:
        return False

    elif contain(query, reasult_set.worst_dist, root):
        leaf_points = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, axis=0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            reasult_set.insert_node(diff[i], root.point_indices[i])
        return False

    elif root.is_leaf:
        assert len(root.point_indices) > 0
        leaf_points = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, axis=0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            reasult_set.insert_node(diff[i], root.point_indices[i])
        return inside(query, reasult_set.worst_dist, root)

    morton_code = 0
    if query[0] > root.center[0]:
        morton_code |= 1
    if query[1] > root.center[1]:
        morton_code |= 2
    if query[2] > root.center[2]:
        morton_code |= 4

    if octree_rnn_search(root.children[morton_code], db, query, reasult_set):
        return True
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        elif not overlap(query, reasult_set.worst_dist, child):
            continue
        elif octree_rnn_search(child, db, query, reasult_set):
            return True
    return inside(query, reasult_set.worst_dist, root)


if __name__ == '__main__':
    db_size = 64000
    leaf_size = 4
    min_extent = 0.0001
    db = np.random.rand(db_size, 3)
    root = octree_construction(db, leaf_size, min_extent)
    traverse_octree(root)

    query = np.array([0, 0, 0])
    k = 5
    knn_result = KNNResultSet(5)
    octree_knn_search(root, db, query, knn_result)
    print(knn_result)
    diff = np.linalg.norm(db - np.expand_dims(query, 0), axis=1)
    sort_indices = np.argsort(diff)
    print(sort_indices[:k])
    print(diff[sort_indices][:k])

    start_t = time.time()
    for i in range(100):
        query = np.random.rand(3)
        rnn_result = RNNResultSet(0.5)
        octree_rnn_search(root, db, query, rnn_result)
        print(rnn_result)
    print('spent time {}s'.format(time.time() - start_t))
