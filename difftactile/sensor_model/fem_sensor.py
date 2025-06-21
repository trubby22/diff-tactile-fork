"""
a class to describe sensor elastomer with FEM
"""

import taichi as ti
import torch
import cv2
from math import pi
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import pickle
from difftactile.sensor_model.fisheye_model import * 

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class FEMDomeSensor:
    def __init__(self, dt=5e-5, sub_steps = 50, init_img_path=None):
        np.set_printoptions(precision=3, floatmode='maxprec', suppress=False)
        self.sub_steps = sub_steps
        self.dt = dt
        self.N_node = 400 # number of nodes in the most inner layer
        self.N_t = 4 # thickness
        self.t_res = 0.2 # [cm]; inter-layer distance
        self.inner_radius = 2.7 # [cm]

        self.all_nodes, self.all_f2v, self.surface_f2v, self.layer_idxs = self.init_mesh()
        self.n_verts = len(self.all_nodes)
        self.n_cells = len(self.all_f2v)
        self.num_triangles = len(self.surface_f2v)

        self.dim = 3
        self.dx = 2 * np.pi * self.inner_radius**2 / self.N_node
        self.vol = self.dx * self.t_res
        # print(f'self.dx: {self.dx}; self.vol: {self.vol}')
        self.rho = 1.145 * 1_000 # density [kg / m^3]
        self.mass = self.vol * self.rho
        self.eps = 1e-10

        self.E_init = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.nu_init = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.E_init[None], self.nu_init[None] = 0.8e4, 0.43#0.3e4, 0.445  # Young's modulus and Poisson's ratio

        self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.lam = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.mu[None] = self.E_init[None] / 2 / (1 + self.nu_init[None])
        self.lam[None] = self.E_init[None] * self.nu_init[None] / (1 + self.nu_init[None]) / (1 - 2 * self.nu_init[None])  # Lame parameters
        self.damping = 10.0

        self.init_x = ti.Vector.field(3, float, self.n_verts, needs_grad=False)
        self.init_x.from_numpy(self.all_nodes.astype(np.float32))
        self.layer_id = ti.field(int, self.n_verts) # indicate layers
        self.layer_id.from_numpy(self.layer_idxs.astype(np.int32))
        self.markers_surface_id_np = np.where(self.layer_idxs==(self.N_t-1))[0]

        self.markers_surface_id = ti.field(int, len(self.markers_surface_id_np))
        self.markers_surface_id.from_numpy(self.markers_surface_id_np.astype(np.int32))
        self.num_surface = len(self.markers_surface_id_np)
        self.surface_cam_loc = ti.Vector.field(2, float, self.num_surface, needs_grad = True)
        self.surface_cam_virtual_loc = ti.Vector.field(2, float, self.num_surface, needs_grad=False)

        # cam model
        self.num_k_closest = 5
        self.initial_markers, interp_idx, interp_weight = self.init_cam_model(init_img_path)
        self.num_markers = len(self.initial_markers)
        self.visualise_2d(interp_idx)

        self.predict_markers = ti.Vector.field(2, float, self.num_markers, needs_grad = True)
        self.virtual_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)

        self.interp_weight = ti.Vector.field(self.num_k_closest, float, self.num_markers, needs_grad=False)
        self.interp_weight.from_numpy(interp_weight.astype(np.float32))
        self.interp_idx = ti.Vector.field(self.num_k_closest, int, self.num_markers)
        self.interp_idx.from_numpy(interp_idx.astype(np.int32))

        self.f2v = ti.Vector.field(4, int, self.n_cells)
        self.f2v.from_numpy(self.all_f2v.astype(np.int32))
        self.contact_seg = ti.Vector.field(3, int, self.num_triangles) # surface triangle mesh
        self.contact_seg.from_numpy(self.surface_f2v.astype(np.int32))

        self.virtual_pos = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad = True)
        self.pos = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad = True)
        self.vel = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad = True)

        self.B = ti.Matrix.field(3, 3, float, self.n_cells, needs_grad=False)
        self.phi = ti.field(float, self.n_cells, needs_grad=False)  # potential energy of each face (Neo-Hookean)

        self.external_force_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_verts), needs_grad = True) # contact force between FEM node to the closest particle
        self.surf_f = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad = True) # surface aggreated 3-axis forces

        # contact model parameters (default)
        self.out_direction = ti.Vector.field(3, float, (), needs_grad=False)

        ## control parameters
        self.d_pos_global = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        self.d_ori_global_euler_angles = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        self.d_pos_local = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        self.d_ori_local = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        self.rot_h = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        self.rot_world = ti.Matrix.field(3, 3, ti.f32, shape = ())
        self.rot_local = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        self.inv_rot = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)

        self.trans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ##
        self.trans_world = ti.Matrix.field(4, 4, ti.f32, shape = ()) ## ee to world
        self.trans_local = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ## ee1 -> ee2
        self.inv_trans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        self.dtrans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        self.control_vel = ti.Vector.field(3, float, shape = (self.n_verts), needs_grad=False)
        self.sdf = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.cache = dict() # for grad backward
        self.first = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.first[None] = True

        self.my_rot_v = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.my_trans_v = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.my_trans_mat = ti.Matrix.field(4, 4, dtype=float, shape=(), needs_grad=False)
        self.my_rot_mat = ti.Matrix.field(3, 3, dtype=float, shape=(), needs_grad=False)

    def init(self, rot_x, rot_y, rot_z, t_dx, t_dy, t_dz):
        rot = R.from_rotvec(np.deg2rad([rot_x, rot_y, rot_z]))
        init_rot = rot.as_matrix()
        trans_mat = np.eye(4)
        trans_mat[0:3,0:3] = init_rot
        trans_mat[0,3] = t_dx; trans_mat[1,3] = t_dy; trans_mat[2,3] = t_dz

        self.rot_local[None] = np.eye(3)
        self.rot_world[None] = init_rot
        self.rot_h[None] = self.rot_world[None] @ self.rot_local[None]
        self.inv_rot[None] = self.rot_h[None].inverse()

        self.trans_local[None] = np.eye(4)
        self.trans_world[None] = trans_mat
        self.trans_h[None] = self.trans_world[None] @ self.trans_local[None]
        self.inv_trans_h[None] = self.trans_local[None].inverse() @ self.trans_world[None].inverse()

        if False and self.first[None]:
            print("\nInitial rotation matrix (init_rot):")
            print(init_rot)
            print("\nInitial transformation matrix (trans_mat):")
            print(trans_mat)
            print("\nLocal rotation matrix (rot_local):")
            print(self.rot_local[None].to_numpy())
            print("\nWorld rotation matrix (rot_world):")
            print(self.rot_world[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.rot_h[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inv_rot[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.trans_local[None].to_numpy())
            print("\nWorld transformation matrix (trans_world):")
            print(self.trans_world[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.trans_h[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inv_trans_h[None].to_numpy())
            print()
        
        self.first[None] = False

        self.init_pos()
        with open(f"output/tactile_sensor.pos.init.pkl", 'wb') as f:
            pickle.dump(self.pos.to_numpy()[0], f)
        np.savetxt(f'output/tactile_sensor.pos.init.csv', self.pos.to_numpy()[0], delimiter=",", fmt='%.2f')
        # with open(f"./output/tactile_sensor.init_x.pkl", 'wb') as f:
        #     pickle.dump(self.init_x.to_numpy(), f)
        # np.savetxt(f'output/tactile_sensor.init_x.csv', self.init_x.to_numpy()[0], delimiter=",", fmt='%.2f')

    def init_cam_model(self, init_img_path=None):
        if init_img_path is None:
            init_img = cv2.imread("./init.png")
        else:
            init_img = cv2.imread(init_img_path)
        initial_markers = get_marker_image(init_img)
        # Overlay initial_markers on the image and save
        overlay_img = init_img.copy()
        for pos in initial_markers:
            center = (int(round(pos[0])), int(round(pos[1])))
            cv2.circle(overlay_img, center, radius=5, color=(0, 0, 255), thickness=2)  # Red circle
        cv2.imwrite("../tasks/output/init_cam_model.png", overlay_img)
        surface_nodes = self.all_nodes[self.markers_surface_id_np]
        cam_3D_nodes = np.array([surface_nodes[:,0], surface_nodes[:,2], surface_nodes[:,1]]).T
        with open(f"output/fem_sensor.cam_3d_nodes.pkl", 'wb') as f:
            pickle.dump(cam_3D_nodes, f)
        np.savetxt(f'output/fem_sensor.cam_3d_nodes.csv', cam_3D_nodes, delimiter=",", fmt='%.2f')
        cam_points = project_points_to_pix(cam_3D_nodes)
        # Overlay cam_points as green circles and save
        cam_points_img = init_img.copy()
        for pt in cam_points:
            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(cam_points_img, center, radius=3, color=(0, 255, 0), thickness=2)  # Green circle
        cv2.imwrite("../tasks/output/cam_points.png", cam_points_img)
        # interpolated markers in 2d & 3d
        interp_idx = []
        interp_weight = []
        surf_2d = []
        for i in range(initial_markers.shape[0]):
            offset = np.linalg.norm(initial_markers[i,0:2] - cam_points,axis=1)
            idx = np.argpartition(offset, self.num_k_closest)
            smallest_idx = idx[:self.num_k_closest]
            inv_offset = 1/offset[smallest_idx]
            total_offset = np.sum(inv_offset)
            weights = inv_offset / total_offset
            loc_2d = np.matmul(cam_points[smallest_idx].T, weights).T
            surf_2d.append(loc_2d)
            interp_idx.append(smallest_idx)
            interp_weight.append(weights)

        surf_2d = np.array(surf_2d)
        interp_idx = np.array(interp_idx)
        interp_weight = np.array(interp_weight)

        # Flatten interp_idx before saving
        interp_idx_flat = interp_idx.flatten()
        with open(f"output/fem_sensor.interp_idx_flat.pkl", 'wb') as f:
            pickle.dump(interp_idx_flat, f)
        with open(f"output/fem_sensor.markers_surface_id_np.pkl", 'wb') as f:
            pickle.dump(self.markers_surface_id_np, f)
        np.savetxt('output/fem_sensor.interp_idx_flat.csv', interp_idx_flat, delimiter=",", fmt='%d')
        np.savetxt('output/fem_sensor.markers_surface_id_np.csv', self.markers_surface_id_np, delimiter=",", fmt='%d')

        return surf_2d, interp_idx, interp_weight

    def visualise_2d(self, interp_idx):
        """
        Project selected 3D marker points to the image plane and overlay them as red dots on './init.png'.
        Save the result as 'output/init-3d-markers.png'.
        """
        # Get the relevant 3D marker points
        marker_indices = np.unique(interp_idx.flatten())
        marker_points_3d = self.all_nodes[self.markers_surface_id_np][marker_indices]
        # Reorder axes to match camera convention
        cam_3D_nodes = np.array([marker_points_3d[:,0], marker_points_3d[:,2], marker_points_3d[:,1]]).T
        # Project to 2D
        marker_points_2d = project_points_to_pix(cam_3D_nodes)

        # Load the image
        img = cv2.imread('./init.png')
        if img is None:
            print('Error: Could not load ./init.png')
            return
        for pt in marker_points_2d:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (x, y), 4, (0,0,255), -1)  # Red dot
        cv2.imwrite('output/init-3d-markers.png', img)

    @ti.kernel
    def extract_markers(self, f:ti.i32):
        for i in range(self.num_surface):
            pos = self.pos[f, self.markers_surface_id[i]]
            init_pos = self.virtual_pos[f, self.markers_surface_id[i]]
            hom_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0])
            hom_init_pos = ti.Vector([init_pos[0], init_pos[1], init_pos[2], 1.0])
            inv_pos = self.inv_trans_h[None] @ hom_pos
            inv_init_pos = self.inv_trans_h[None] @ hom_init_pos
            cam_pos = ti.Vector([inv_pos[0], inv_pos[2], inv_pos[1]])

            cam_init_pos = ti.Vector([inv_init_pos[0], inv_init_pos[2], inv_init_pos[1]])
            cam_loc = project_3d_2d(cam_pos)
            cam_init_loc = project_3d_2d(cam_init_pos)
            self.surface_cam_loc[i] = cam_loc
            self.surface_cam_virtual_loc[i] = cam_init_loc

        for i in range(self.num_markers):
            smallest_idx = self.interp_idx[i]
            weights = self.interp_weight[i]
            loc_2d = ti.Vector([0.0, 0.0])
            init_loc_2d = ti.Vector([0.0, 0.0])
            for j in range(self.num_k_closest):
                loc_2d += weights[j] * self.surface_cam_loc[smallest_idx[j]]
                init_loc_2d += weights[j] * self.surface_cam_virtual_loc[smallest_idx[j]]
            self.predict_markers[i] = loc_2d
            self.virtual_markers[i] = init_loc_2d

    @ti.kernel
    def set_material_params(self, E:ti.f32, nu:ti.f32):
        self.E_init[None], self.nu_init[None] = E, nu # Young's modulus and Poisson's ratio
        self.mu[None] = self.E_init[None] / 2 / (1 + self.nu_init[None])
        self.lam[None] = self.E_init[None] * self.nu_init[None] / (1 + self.nu_init[None]) / (1 - 2 * self.nu_init[None])  # Lame parameters

    @ti.func
    def eul2mat(self, rot_v, trans_v):
        # rot_v: euler angles (degrees) for rotation (x,y,z)
        # trans_v: translation (x,y,z)
        rot_v_r = ti.math.radians(rot_v)
        rot_x = rot_v_r[0]
        rot_y = rot_v_r[1]
        rot_z = rot_v_r[2]
        mat_x = ti.Matrix([[1.0, 0.0, 0.0],[0.0, ti.cos(rot_x), -ti.sin(rot_x)],[0.0, ti.sin(rot_x), ti.cos(rot_x)]])
        mat_y = ti.Matrix([[ti.cos(rot_y), 0.0, ti.sin(rot_y)],[0.0, 1.0, 0.0],[-ti.sin(rot_y), 0.0, ti.cos(rot_y)]])
        mat_z = ti.Matrix([[ti.cos(rot_z), -ti.sin(rot_z), 0.0],[ti.sin(rot_z), ti.cos(rot_z), 0.0],[0.0, 0.0, 1.0]])
        mat_R = mat_z @ mat_y @ mat_x
        trans_h = ti.Matrix.identity(float, 4)
        trans_h[0:3, 0:3] = mat_R
        trans_h[0:3, 3] = trans_v
        return trans_h, mat_R

    @ti.kernel
    def set_vel(self, f:ti.i32):
        for p in range(self.n_verts):
            self.vel[f, p] = self.control_vel[p]

    @ti.kernel
    def set_pose_control(self):
        # this is in local coord
        self.d_pos_local[None] = self.inv_rot[None] @ self.d_pos_global[None]
        self.d_ori_local[None] = self.inv_rot[None] @ self.d_ori_global_euler_angles[None]

        self.my_rot_v[None] = self.d_ori_local[None] * self.dt * (self.sub_steps -1)
        self.my_trans_v[None] = self.d_pos_local[None] * self.dt * (self.sub_steps -1)
        self.my_trans_mat[None], self.my_rot_mat[None] = self.eul2mat(self.my_rot_v[None], self.my_trans_v[None])

        self.dtrans_h[None] = self.trans_world[None] @ self.my_trans_mat[None] @ (self.trans_world[None].inverse())

        self.trans_local[None] = self.my_trans_mat[None] @ self.trans_local[None]
        self.trans_h[None] = self.trans_world[None] @ self.trans_local[None]
        self.inv_trans_h[None] = self.trans_h[None].inverse()

        self.rot_local[None] = self.my_rot_mat[None] @ self.rot_local[None]
        self.rot_h[None] = self.rot_world[None] @ self.rot_local[None]
        self.inv_rot[None] = self.rot_h[None].inverse()

        if False:
            print(f'self.d_pos_global[None]: {self.d_pos_global[None]}')
            print(f'self.d_pos_local[None]: {self.d_pos_local[None]}')
            print(f'target angular speed (rotation matrix): self.my_rot_mat[None]: {self.my_rot_mat[None]}')
            print()

    def set_pose_control_maybe_print(self):
        if False:
            print("\nInput rotation vector (self.my_rot_v[None]):")
            print(self.my_rot_v[None].to_numpy())
            print("\nInput translation vector (self.my_trans_v[None]):")
            print(self.my_trans_v[None].to_numpy())
            print("\nComputed transformation matrix (self.my_trans_mat[None]):")
            print(self.my_trans_mat[None].to_numpy())
            print("\nComputed rotation matrix (self.my_rot_mat[None]):")
            print(self.my_rot_mat[None].to_numpy())
            print("\nDelta transformation matrix (dtrans_h):")
            print(self.dtrans_h[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.trans_local[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.trans_h[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inv_trans_h[None].to_numpy())
            print("\nLocal rotation matrix (rot_local):")
            print(self.rot_local[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.rot_h[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inv_rot[None].to_numpy())
            print()

    @ti.kernel
    def set_pose_control_bp(self):

        rot_v = self.d_ori_local[None] * self.dt * (self.sub_steps -1)
        trans_v = self.d_pos_local[None] * self.dt * (self.sub_steps -1)
        trans_mat, rot_mat = self.eul2mat(rot_v, trans_v)
        self.dtrans_h[None] = self.trans_world[None] @ trans_mat @ (self.trans_world[None].inverse())

        self.inv_trans_h[None] = self.trans_h[None].inverse()
        self.inv_rot[None] = self.rot_h[None].inverse()

    @ti.kernel
    def set_control_vel(self, f:ti.i32):
        for i in range(self.n_verts):
            init_t_pos = self.virtual_pos[f, i]
            after_t_pos = self.dtrans_h[None] @ ti.Vector([init_t_pos[0], init_t_pos[1], init_t_pos[2], 1.0]) # 4 x 1 homogeneous
            self.control_vel[i][0] = (after_t_pos[0] - init_t_pos[0]) / (self.dt * (self.sub_steps -1))
            self.control_vel[i][1] = (after_t_pos[1] - init_t_pos[1]) / (self.dt * (self.sub_steps -1))
            self.control_vel[i][2] = (after_t_pos[2] - init_t_pos[2]) / (self.dt * (self.sub_steps -1))

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for k in range(self.num_triangles):
            a, b, c = self.contact_seg[k]
            self.surf_f[f] += 1/3 * self.external_force_field[f,a] * self.dx
            self.surf_f[f] += 1/3 * self.external_force_field[f,b] * self.dx
            self.surf_f[f] += 1/3 * self.external_force_field[f,c] * self.dx

    @ti.func
    def get_euler_angles(self) -> ti.types.vector(3, ti.f32):
        # Extract Euler angles from rotation matrix
        rot_mat = self.rot_h[None]
        
        # Extract roll (x-axis rotation)
        roll = ti.math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        
        # Extract pitch (y-axis rotation)
        pitch = ti.math.atan2(-rot_mat[2, 0], ti.sqrt(rot_mat[2, 1] * rot_mat[2, 1] + rot_mat[2, 2] * rot_mat[2, 2]))
        
        # Extract yaw (z-axis rotation)
        yaw = ti.math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        
        # Convert to degrees
        roll_deg = ti.math.degrees(roll)
        pitch_deg = ti.math.degrees(pitch)
        yaw_deg = ti.math.degrees(yaw)
        
        euler_angles = ti.Vector([roll_deg, pitch_deg, yaw_deg])

        if False:
            print(f'rotation matrix (self.rot_h[None]): {self.rot_h[None]}')
            print(f'euler angles (euler_angles): {euler_angles}')
            print()

        return euler_angles

    def fibonacci_sphere(self, samples=100, scale=1.0):
        # sample points evenly on a hemisphere
        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians
        idx = np.arange(samples).astype(float)
        upper_y = 1.0
        lower_y = 0.0
        y = lower_y + (idx / (samples - 1)) * (upper_y - lower_y)
        radius = np.sqrt(1 - y * y)
        theta = phi * idx
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        stem_wall_height = upper_y - 0.6 / scale
        circle_r = np.sqrt(1.0 ** 2 - stem_wall_height ** 2)

        hemisphere_points = np.vstack((x, y, z)).T
        points = hemisphere_points.copy()

        stem_wall = points[:, 1] < stem_wall_height
        # Compute angle in xz-plane for these points
        angles = np.arctan2(points[stem_wall, 2], points[stem_wall, 0])
        # Map xz to circle of radius circle_r
        points[stem_wall, 0] = circle_r * np.cos(angles)
        points[stem_wall, 2] = circle_r * np.sin(angles)
        # y remains unchanged

        hemisphere_points *= scale
        points *= scale

        return points, stem_wall_height, stem_wall, hemisphere_points

    def generate_cylinder_lateral_surface(self, samples=100, scale=1.0):
        """
        Generate evenly distributed points on the lateral surface of a cylinder using the Fibonacci method.
        The cylinder is centered at the origin, aligned along the y-axis, with height from -0.5 to 0.5 (before scaling).
        The radius is 0.5 (before scaling).
        """
        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians
        idx = np.arange(samples).astype(float)
        # y goes from (scale - 0.6 - 1.0) to (scale - 0.6)
        upper_y = (scale - 0.6) / scale
        lower_y = (scale - 0.6 - 1.0) / scale
        y = lower_y + (idx / (samples - 1)) * (upper_y - lower_y)
        theta = phi * idx
        x = 0.5 * np.cos(theta)
        z = 0.5 * np.sin(theta)

        points = scale * np.vstack((x, y, z)).T
        return points

    def init_mesh(self):
        ## a hemisphere sensor elastomer centered around [0,0,0]
        all_nodes = []
        surface_f2v = None
        layer_idxs = []
        layer_height = []
        num_cur_node = 0
        for i in range(self.N_t):
            rad = self.inner_radius + i * self.t_res
            ratio = (rad**2) / (self.inner_radius**2)
            n_node = int(self.N_node * ratio)
            layer_nodes, cylinder_height, is_stem_wall, hemisphere_nodes = self.fibonacci_sphere(samples=n_node, scale = rad)
            if i == 0:
                min_cylinder_height = cylinder_height
            all_nodes.append(layer_nodes)
            cur_layer_height = np.copy(layer_nodes[:, 1])
            cur_layer_height[~is_stem_wall] = cylinder_height + 100 + i * 100
            layer_height.append(cur_layer_height)
            layer_idxs.append([i] * n_node)
            if i == self.N_t-1:
                x = hemisphere_nodes[:,0]
                y = hemisphere_nodes[:,1]
                z = hemisphere_nodes[:,2]
                sphere_points = np.vstack((x, z)).T
                surface_f2v = num_cur_node + Delaunay(sphere_points).simplices
            num_cur_node += n_node
        layer_height = np.concatenate(layer_height, axis=0)
        all_nodes = np.concatenate(all_nodes,axis=0)
        point_a_idx = np.argmax(all_nodes[:, 1])
        point_a = all_nodes[point_a_idx]
        delta_y = 2.4 - point_a[1]
        translation_vec = np.array([0, delta_y, 0])
        all_nodes += translation_vec
        triangle_nodes = np.array([all_nodes[:,0], layer_height, all_nodes[:,2]]).T
        inner_volume_particles = np.zeros((100, 3), dtype=np.float32)
        inner_volume_particles[:, 1] = np.linspace(-1 * min_cylinder_height * 0.9, min_cylinder_height * 0.9, 100)
        triangle_nodes = np.vstack([triangle_nodes, inner_volume_particles])

        all_f2v = Delaunay(triangle_nodes).simplices
        layer_idxs = np.concatenate(layer_idxs,axis=0)

        # --- Filter out any simplex containing an inner_volume_particle ---
        n_total = triangle_nodes.shape[0]
        n_inner_particles = inner_volume_particles.shape[0]
        first_inner_idx = n_total - n_inner_particles
        # Keep only those simplices where all indices are < first_inner_idx
        mask_no_inner_particles = np.all(all_f2v < first_inner_idx, axis=1)
        all_f2v = all_f2v[mask_no_inner_particles]
        # --- End filter ---

        if False:
            # --- Begin filtering out inner-most layer (layer 0) ---
            # Find indices of inner-most layer nodes
            inner_layer_mask = (layer_idxs == 0)
            inner_layer_indices = np.where(inner_layer_mask)[0]
            n_inner = len(inner_layer_indices)
            # Indices to keep (not in inner-most layer)
            keep_mask = ~inner_layer_mask
            keep_indices = np.where(keep_mask)[0]
            # Build mapping from old indices to new indices
            old_to_new = -np.ones(len(layer_idxs), dtype=int)
            old_to_new[keep_indices] = np.arange(len(keep_indices))
            # Filter all_nodes and layer_idxs
            all_nodes = all_nodes[keep_mask]
            layer_idxs = layer_idxs[keep_mask]
            # Filter all_f2v: keep only those tetrahedra whose all nodes are not in inner-most layer
            mask_f2v = np.all(np.isin(all_f2v, keep_indices), axis=1)
            all_f2v = all_f2v[mask_f2v]
            # Remap indices in all_f2v
            all_f2v = old_to_new[all_f2v]
            # Filter surface_f2v and remap indices (subtract n_inner)
            if surface_f2v is not None:
                # Only keep triangles whose all nodes are not in inner-most layer
                mask_surf = np.all(np.isin(surface_f2v, keep_indices), axis=1)
                surface_f2v = surface_f2v[mask_surf]
                surface_f2v = old_to_new[surface_f2v]
            triangle_nodes = triangle_nodes[keep_mask]
            # --- End filtering ---

        pickles = [
            ('all_nodes', all_nodes),
            ('triangle_nodes', triangle_nodes),
            ('all_f2v', all_f2v),
            ('layer_idxs', layer_idxs),
        ]
        for x, y in pickles:
            with open(f"output/tactile_sensor.{x}.pkl", 'wb') as f:
                pickle.dump(y, f)
            np.savetxt(f'output/tactile_sensor.{x}.csv', y, delimiter=",", fmt='%.2f')
        return all_nodes, all_f2v, surface_f2v, layer_idxs

    @ti.kernel
    def init_pos(self):
        for idx in range(self.n_verts):
            before_t_pos = self.init_x[idx] # before any world transformation
            after_t_pos = self.trans_h[None] @ ti.Vector([before_t_pos[0], before_t_pos[1], before_t_pos[2], 1.0]) # 4 x 1 homogeneous

            self.pos[0, idx] = ti.Vector([after_t_pos[0], after_t_pos[1], after_t_pos[2]])
            # reset init x to track the whole body movement
            self.virtual_pos[0, idx] = self.pos[0, idx]

        for i in range(self.n_cells):
            ia, ib, ic, id = self.f2v[i]
            a, b, c, d = self.pos[0, ia], self.pos[0, ib], self.pos[0, ic], self.pos[0, id]
            B_i_inv = ti.Matrix.cols([a - d, b - d, c - d])
            self.B[i] = B_i_inv.inverse()

        self.out_direction[None] = self.rot_h[None] @ ti.Vector([0.0, 1.0, 0.0])

    @ti.func
    def find_closest(self, grid_p, f):
        cur_min_offset = 100.0 # arbitrary large value
        cur_min_idx = -1
        for k in range(self.num_triangles):
            a, b, c = self.contact_seg[k]
            p_1 = self.pos[f, a] # triangle's 1st node
            p_2 = self.pos[f, b] # triangle's 2nd node
            p_3 = self.pos[f, c] # triangle's 3rd node
            p_c = 1/3 * (p_1 + p_2 + p_3) # center of the segment
            offset_p = (p_c - grid_p).norm(self.eps) # distance to the center point of the segment

            if (offset_p < cur_min_offset):
                cur_min_offset = offset_p
                cur_min_idx = k

        return cur_min_idx

    @ti.func
    def find_sdf(self, grid_p, grid_v, min_idx, f, offset = 0.0):
        a, b, c = self.contact_seg[min_idx]
        p_1 = self.pos[f, a]
        p_2 = self.pos[f, b]
        p_3 = self.pos[f, c]

        normal_plane = ti.math.cross(p_2-p_1, p_3-p_1) # plane's norm
        normal_plane = normal_plane.normalized(self.eps)
        sign_p1 = ti.math.sign(normal_plane.dot(self.out_direction[None]))
        normal_plane = sign_p1 * normal_plane # facing up

        d_s = grid_p - p_1 # vector from the first node to the particle
        sdf = d_s.dot(normal_plane) # distance to the plane
        project_p1 = grid_p - sdf * normal_plane # projection of point on the segment
        norm_v = -1* normal_plane

        v0 = p_3 - p_1
        v1 = p_2 - p_1
        v2 = project_p1 - p_1
        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1)
        dot02 = v0.dot(v2)
        dot11 = v1.dot(v1)
        dot12 = v1.dot(v2)
        inv_d = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_d
        v = (dot00 * dot12 - dot01 * dot02) * inv_d

        ### correct with an offset for pbd collision
        sdf -= offset
        relative_v = grid_v - 1/3 * (self.vel[f, a] + self.vel[f, b] + self.vel[f, c])
        contact_flag = sdf < 0 and u >= 0 and v >= 0.0 and (u + v <= 1)
        return sdf, norm_v, relative_v, contact_flag

    @ti.kernel
    def reset_contact(self):
        self.external_force_field.fill(0.0)
        self.surf_f.fill(0.0)

    @ti.func
    def update_contact_force(self, min_idx, ext_v1, f):
        a, b, c = self.contact_seg[min_idx]
        self.external_force_field[f, a] += 1/3 * ext_v1
        self.external_force_field[f, b] += 1/3 * ext_v1
        self.external_force_field[f, c] += 1/3 * ext_v1

    @ti.kernel
    def update(self, f:ti.i32):

        for i in range(self.n_cells):
            ia, ib, ic, id = self.f2v[i]
            a, b, c, d = self.pos[f, ia], self.pos[f, ib], self.pos[f, ic], self.pos[f, id]
            D_i = ti.Matrix.cols([a - d, b - d, c - d])
            V_i = ti.abs(D_i.determinant()) / 6
            F_i = D_i @ self.B[i]

            ## stable neo-hooken
            J = F_i.determinant()
            IC = (F_i.transpose() @ F_i).trace()
            dJdF0 = F_i[:,1].cross(F_i[:,2])
            dJdF1 = F_i[:,2].cross(F_i[:,0])
            dJdF2 = F_i[:,0].cross(F_i[:,1])
            dJdF = ti.Matrix.cols([dJdF0, dJdF1, dJdF2])
            alpha = 1 + 0.75 * self.mu[None]/self.lam[None]
            stress = self.mu[None] * (1 - 1/(IC+1)) * F_i + self.lam[None] * (J - alpha) * dJdF

            H = -V_i * stress @ self.B[i].transpose()
            verts = ti.Vector([ia, ib, ic, id])
            for k in ti.static(range(3)):
                force = ti.Vector([H[j,k] for j in range(3)])
                self.vel[f,verts[k]] += self.dt * force / self.mass
                self.vel[f,verts[3]] += -1*self.dt * force / self.mass


    @ti.kernel
    def update2(self, f:ti.i32):
        for i in range(self.n_verts):
            v_out = ti.Vector([0.0, 0.0, 0.0])
            v_out += self.vel[f,i]
            v_out += self.dt * self.external_force_field[f,i] / self.mass

            ### stick the bottom layer to be fixed
            cond = self.layer_id[i] == 0
            if cond:
                v_out = self.control_vel[i]
            self.vel[f+1, i] = v_out
            self.pos[f+1, i] = self.pos[f, i] + self.dt * v_out
            # update virtual pos
            self.virtual_pos[f+1, i] = self.virtual_pos[f, i] + self.dt * self.control_vel[i]


    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_verts):
            self.pos[target, p] = self.pos[source, p]
            self.vel[target, p] = self.vel[source, p]
            self.virtual_pos[target, p] = self.virtual_pos[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_verts):
            self.pos.grad[target, p] = self.pos.grad[source, p]
            self.vel.grad[target, p] = self.vel.grad[source, p]
            self.virtual_pos.grad[target, p] = self.virtual_pos.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                self.trans_h[None][j,k] = cache_trans[j,k]
        for j in range(3):
            for k in range(3):
                self.rot_h[None][j,k] = cache_rot[j,k]
        for p in range(self.n_verts):
            for i in ti.static(range(self.dim)):
                self.pos[f, p][i] = cache_pos[p,i]
                self.vel[f, p][i] = cache_vel[p,i]
                self.virtual_pos[f, p][i] = cache_virtual_pos[p, i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                self.predict_markers[p][i] = cache_predict_markers[p,i]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                cache_trans[j,k] = self.trans_h[None][j,k]
        for j in range(3):
            for k in range(3):
                cache_rot[j,k] = self.rot_h[None][j,k]
        for p in range(self.n_verts):
            for i in ti.static(range(self.dim)):
                cache_pos[p,i] = self.pos[f, p][i]
                cache_vel[p,i] = self.vel[f, p][i]
                cache_virtual_pos[p, i] = self.virtual_pos[f, p][i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                cache_predict_markers[p,i] = self.predict_markers[p][i]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['pos'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['vel'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['trans_h'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['rot_h'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['virtual_pos'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['predict_markers'] = torch.zeros((self.num_markers, 2), dtype=TC_TYPE, device=device)
        self.add_step_to_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'], self.cache[cur_step_name]['trans_h'], self.cache[cur_step_name]['virtual_pos'], self.cache[cur_step_name]['rot_h'], self.cache[cur_step_name]['predict_markers'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'], self.cache[cur_step_name]['trans_h'], self.cache[cur_step_name]['virtual_pos'], self.cache[cur_step_name]['rot_h'], self.cache[cur_step_name]['predict_markers'])

    @ti.kernel
    def clear_loss_grad(self):
        self.E_init.grad[None] = 0.0
        self.nu_init.grad[None] = 0.0
        self.mu.grad[None] = 0.0
        self.lam.grad[None] = 0.0
        self.predict_markers.grad.fill(0.0)
        self.surface_cam_loc.grad.fill(0.0)
        self.d_pos_local.grad[None].fill(0.0)
        self.d_ori_local.grad[None].fill(0.0)
        self.rot_h.grad[None].fill(0.0)
        self.rot_local.grad[None].fill(0.0)
        self.inv_rot.grad[None].fill(0.0)
        self.trans_h.grad[None].fill(0.0)
        self.trans_local.grad[None].fill(0.0)
        self.inv_trans_h.grad[None].fill(0.0)
        self.dtrans_h.grad[None].fill(0.0)
        self.control_vel.grad.fill(0.0)

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.surf_f.grad.fill(0.0)
        self.external_force_field.grad.fill(0.0)
        for p in range(self.n_verts):
            for t in range(f):
                self.pos.grad[t, p].fill(0.0)
                self.vel.grad[t, p].fill(0.0)
                self.virtual_pos.grad[t, p].fill(0.0)

    def get_min_z_from_cache(self, t):
        """Returns the minimum z-coordinate from the cached positions at timestep t.
        
        Args:
            t (int): The timestep to get the minimum z-coordinate from
            
        Returns:
            float: The minimum z-coordinate across all vertices
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get the z-coordinates (third column) and find minimum
        z_coords = self.cache[cur_step_name]['pos'][:, 2]
        return float(z_coords.min().item())

    def get_min_z_ix(self, t):
        """Returns the minimum z-coordinate from the cached positions at timestep t.
        
        Args:
            t (int): The timestep to get the minimum z-coordinate from
            
        Returns:
            float: The minimum z-coordinate across all vertices
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get the z-coordinates (third column) and find minimum
        z_coords = self.cache[cur_step_name]['pos'][:, 2]
        return z_coords.argmin().item()

    def get_high_z_max_y_ix(self, t):
        """Returns the index of the point that has maximum y-coordinate among points
        whose z-coordinate is within 0.2 of the maximum z value.
        
        Args:
            t (int): The timestep to get the point index from
            
        Returns:
            int: The index of the vertex that meets the criteria
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get all positions
        positions = self.cache[cur_step_name]['pos']
        
        # Get z-coordinates and find maximum
        z_coords = positions[:, 2]
        max_z = float(z_coords.max().item())
        
        # Create mask for points within 0.2 of max z
        z_mask = (z_coords >= (max_z - 0.2))
        
        # Get y-coordinates of points that meet z criteria
        y_coords = positions[:, 1]
        y_coords_filtered = y_coords.clone()
        y_coords_filtered[~z_mask] = float('-inf')  # Set non-matching points to negative infinity
        
        # Find index of maximum y among filtered points
        return int(y_coords_filtered.argmax().item())

    def get_xyz_angle_from_cache(self, t, idx):
        """Returns the x and y coordinates of a point from the cached positions at timestep t,
        and computes the angle this point forms with a reference point, assuming the point lies
        on a circle centered at the reference point.
        
        Args:
            t (int): The timestep to get the coordinates from
            idx (int): The index of the point to get coordinates for
            ref_x (float): x-coordinate of the reference point (center of circle)
            ref_y (float): y-coordinate of the reference point (center of circle)
            
        Returns:
            tuple: (x, y, angle) where:
                - x, y are the coordinates as floats
                - angle is in degrees, measured counterclockwise from positive x-axis
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        if idx < 0 or idx >= self.n_verts:
            raise ValueError(f"Index {idx} out of bounds. Must be between 0 and {self.n_verts-1}")
            
        # Get the x and y coordinates (first and second columns)
        point = self.cache[cur_step_name]['pos'][idx]
        x = float(point[0].item())
        y = float(point[1].item())
        z = float(point[2].item())
        
        # Compute angle using atan2 and convert to degrees
        dx = x - 12.50
        dy = y - 11.50
        angle_rad = np.arctan2(dy, dx)
        
        return (x, y, z), angle_rad

    def get_keypoint_indices(self, f: ti.i32):
        # Convert positions to numpy array
        positions = self.pos.to_numpy()[f]
        
        # Point A: minimum z coordinate
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmin(z_coords))
        
        # Points B and C: high z coordinate points
        max_z = float(np.max(z_coords))
        z_mask = (z_coords >= (max_z - 0.2))
        
        # Point B: max x coordinate among high z points
        x_coords = positions[:, 0]
        x_coords_filtered = x_coords.copy()
        x_coords_filtered[~z_mask] = float('-inf')
        point_b_idx = int(np.argmax(x_coords_filtered))
        
        # Point C: max y coordinate among high z points
        y_coords = positions[:, 1]
        y_coords_filtered = y_coords.copy()
        y_coords_filtered[~z_mask] = float('-inf')
        point_c_idx = int(np.argmax(y_coords_filtered))
        
        return np.array([point_a_idx, point_b_idx, point_c_idx])

    def get_keypoint_indices_numpy_point_a(self):
        # Convert positions to numpy array
        positions = self.all_nodes
        
        # Point A: max y coordinate
        y_coords = positions[:, 1]
        point_a_idx = int(np.argmax(y_coords))

        return point_a_idx

    def get_keypoint_coordinates(self, f: int, keypoint_indices: np.ndarray) -> np.ndarray:
        """
        Get coordinates for specified keypoint indices at a given frame.
        
        Args:
            f: Frame index
            keypoint_indices: Array of keypoint indices to get coordinates for
            
        Returns:
            numpy array of shape (num_points, 3) containing the coordinates
        """
        # Convert positions to numpy array for the given frame
        positions = self.pos.to_numpy()[f]
        
        # Extract coordinates for the specified indices
        coordinates = positions[keypoint_indices]
        
        return coordinates
