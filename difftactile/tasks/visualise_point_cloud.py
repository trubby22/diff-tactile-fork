import numpy as np
import open3d as o3d
import pickle

with open(f'output/tactile_sensor.all_nodes.pkl', 'rb') as f:
    points = pickle.load(f)
with open(f'output/tactile_sensor.all_f2v.pkl', 'rb') as f:
    tetrahedra_indices = pickle.load(f)

tetrahedra_indices = tetrahedra_indices.astype(int)

all_triangle_faces = []
for tetra in tetrahedra_indices:
    v0, v1, v2, v3 = tetra
    all_triangle_faces.append([v0, v1, v2])
    all_triangle_faces.append([v0, v1, v3])
    all_triangle_faces.append([v0, v2, v3])
    all_triangle_faces.append([v1, v2, v3])

triangle_faces_np = np.array(all_triangle_faces, dtype=np.int32)

if False:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes])

if True:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangle_faces_np)
    mesh.compute_vertex_normals()
    mesh = mesh.remove_duplicated_triangles()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
