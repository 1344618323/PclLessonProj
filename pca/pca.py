import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud


def pca(data: np.ndarray):
    avg_point = np.mean(data, axis=0)
    m = np.zeros(shape=(3, 3))
    for i in range(data.shape[0]):
        pi = (data[i] - avg_point).reshape(3, 1)
        m += pi @ pi.transpose()
    eig_val, eig_vec = np.linalg.eig(m)
    sort_indices = np.argsort(eig_val)
    return eig_val[sort_indices], eig_vec[:, sort_indices]


def main():
    point_cloud_pynt = PyntCloud.from_file('/home/cxn/code/ModelNet40/ModelNet40_PLY/airplane/train/airplane_0001.ply')
    point_cloud_o3d = point_cloud_pynt.to_instance('open3d', mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    points = point_cloud_pynt.points.to_numpy()
    print('total points number is: {}'.format(points.shape[0]))
    v, vec = pca(points)
    print('main_vec: {}'.format(vec[:, 2]))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(
        [np.mean(points, axis=0), np.mean(points, axis=0) + 100 * vec[:, 2]])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
    # o3d.visualization.draw_geometries([point_cloud_o3d, line_set])


if __name__ == '__main__':
    main()
