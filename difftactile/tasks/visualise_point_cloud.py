import numpy as np
import open3d as o3d
import pickle

with open(f'output/tactile_sensor.all_nodes.pkl', 'rb') as f:
    all_nodes = pickle.load(f)

with open(f'output/fem_sensor.interp_idx_flat.pkl', 'rb') as f:
    interp_idx_flat = pickle.load(f)

with open(f'output/fem_sensor.surface_id_np.pkl', 'rb') as f:
    surface_id_np = pickle.load(f)

surface_nodes = all_nodes[surface_id_np]

if True:
    interp_idx_flat = np.unique(interp_idx_flat)
    all_indices = np.arange(surface_nodes.shape[0])
    mask_yellow = np.zeros(surface_nodes.shape[0], dtype=bool)
    mask_yellow[interp_idx_flat] = True
    mask_blue = ~mask_yellow

    # Yellow surface_nodes (selected)
    pcd_yellow = o3d.geometry.PointCloud()
    pcd_yellow.surface_nodes = o3d.utility.Vector3dVector(surface_nodes[mask_yellow])
    yellow_color = np.array([[1.0, 1.0, 0.0]] * np.sum(mask_yellow))
    pcd_yellow.colors = o3d.utility.Vector3dVector(yellow_color)

    # Blue surface_nodes (remaining)
    pcd_blue = o3d.geometry.PointCloud()
    pcd_blue.surface_nodes = o3d.utility.Vector3dVector(surface_nodes[mask_blue])
    blue_color = np.array([[0.0, 0.0, 1.0]] * np.sum(mask_blue))
    pcd_blue.colors = o3d.utility.Vector3dVector(blue_color)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75, origin=[0, 0, 0])

    # Visualize with different sizes
    o3d.visualization.draw_geometries(
        [pcd_blue, pcd_yellow, axes],
        point_show_normal=False,
        point_size=0.75,
        window_name="Point Cloud Visualization",
        width=800,
        height=600,
        left=50,
        top=50,
        mesh_show_back_face=False,
        render_option_callback=lambda vis: (
            vis.get_render_option().point_size = 0.75,
            vis.get_render_option().background_color = np.array([0,0,0]),
            vis.get_render_option().show_coordinate_frame = True,
            vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        )
    )
    # Now draw yellow surface_nodes with larger size
    o3d.visualization.draw_geometries(
        [pcd_yellow, axes],
        point_show_normal=False,
        point_size=3.0,
        window_name="Highlighted surface_nodes",
        width=800,
        height=600,
        left=900,
        top=50,
        mesh_show_back_face=False,
        render_option_callback=lambda vis: (
            vis.get_render_option().point_size = 3.0,
            vis.get_render_option().background_color = np.array([0,0,0]),
            vis.get_render_option().show_coordinate_frame = True,
            vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        )
    )

if False:
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

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(surface_nodes)
    mesh.triangles = o3d.utility.Vector3iVector(triangle_faces_np)
    mesh.compute_vertex_normals()
    mesh = mesh.remove_duplicated_triangles()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
