import numpy as np
import open3d as o3d
import pickle

# n = 10000
# points = np.random.rand(n, 3) * 10 - 5 # Random points between -5 and 5

with open(f'output/tactile_sensor.layer_nodes_3.pkl', 'rb') as f:
    points = pickle.load(f)

# print(type(points))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, axes])
