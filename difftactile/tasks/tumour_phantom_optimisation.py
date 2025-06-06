from difftactile.tasks.tumour_phantom_visualisation import (
    ContactVisualisation,
    set_up_gui,
    update_gui,
)
from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.mpm_elastic import MPMObj

import taichi as ti
import numpy as np
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

off_screen = False
enable_gui1 = True
enable_gui2 = True
enable_gui3 = True
USE_TACTILE = True
USE_STATE = True


@ti.data_oriented
class Contact(ContactVisualisation):
    def __init__(
        self,
        use_tactile=True,
        use_state=True,
        dt=5e-5,
        num_frames=300,
        num_sub_frames=50,
        obj=None,
    ):
        super().__init__()

        self.dt = dt
        self.num_frames = num_frames
        self.num_sub_frames = num_sub_frames
        self.fem_sensor1 = FEMDomeSensor(dt, num_sub_frames)
        self.use_tactile = use_tactile
        self.use_state = use_state
        self.mpm_object = MPMObj(
            dt=dt,
            num_sub_frames=num_sub_frames,
            obj_name=obj,
            space_scale=self.space_scale,
            obj_scale=self.obj_scale,
            density=1.50 * 2,
            rho=0.3,
        )
        self.alpha = ti.field(float, ())
        self.beta = ti.field(float, ())
        self.alpha[None] = 1e1
        self.beta[None] = 5e-12
        self.num_sensor = 1
        self.init()
        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=True)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kn[None] = 34.53
        self.kd[None] = 269.44
        self.kt[None] = 108.72
        self.friction_coeff[None] = 14.16
        self.fem_sensor1.mu[None] = 1294.01
        self.fem_sensor1.lam[None] = 9201.11
        self.contact_idx = ti.Vector.field(
            self.num_sensor,
            dtype=int,
            shape=(
                self.num_sub_frames,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
            ),
        )
        self.p_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=True
        )
        self.o_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=True
        )
        self.loss = ti.field(float, (), needs_grad=True)
        self.angle = ti.field(float, (self.num_frames), needs_grad=True)
        self.contact_detect_flag = ti.field(float, (), needs_grad=True)
        self.predict_force1 = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.contact_force1 = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.init_visualisation()
        self.norm_eps = 1e-11

        self.set_up_target_marker_positions()

        self.init_visualisation()
        self.norm_eps = 1e-11

    def init(self):
        x = 3.0
        y = 2.25
        z = 3.0

        self.phantom_pos = [x, y, z]
        self.phantom_ori = [0.0, 0.0, 0.0]
        self.phantom_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(self.phantom_pos, self.phantom_ori, self.phantom_vel)
        rx1 = 180.0
        ry1 = 0.0
        rz1 = 0.0
        t_dx1 = x - 0.5
        t_dy1 = y + self.fem_sensor1.outer_radius + 3.0
        t_dz1 = z
        self.fem_sensor1.init(rx1, ry1, rz1, t_dx1, t_dy1, t_dz1)

    @ti.kernel
    def init_pos_control(self):
        sf = 10
        init_steps = 500

        vx1 = 0.0
        vy1 = 0.0
        vz1 = 0.0
        rx1 = 0.0
        ry1 = 0.0
        rz1 = 0.0
        for i in range(0, init_steps):
            self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
            self.o_sensor1[i] = ti.Vector([rx1, ry1, rz1])

        if True:
            vx1 = 0.0
            vy1 = 1.5 * sf
            vz1 = 0.0
            rx1 = 0.0
            ry1 = 0.0
            rz1 = 0.0
            for i in range(init_steps, self.num_frames):
                self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
                self.o_sensor1[i] = ti.Vector([rx1, ry1, rz1])

    @ti.kernel
    def set_pos_control(self, f: ti.i32):
        self.fem_sensor1.d_pos[None] = self.p_sensor1[f]
        self.fem_sensor1.d_ori[None] = self.o_sensor1[f]

    def update(self, f):
        self.mpm_object.compute_new_F(f)
        self.mpm_object.svd(f)
        self.mpm_object.p2g(f)
        self.fem_sensor1.update(f)
        self.mpm_object.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.mpm_object.grid_op(f)
        self.mpm_object.g2p(f)
        self.fem_sensor1.update2(f)

    def update_grad(self, f):
        self.fem_sensor1.update2.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.fem_sensor1.update.grad(f)
        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

    @ti.kernel
    def clear_state_loss_grad(self):
        self.angle.fill(0.0)
        self.angle.grad.fill(0.0)

    @ti.kernel
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0
        self.contact_detect_flag.grad[None] = 0.0
        self.contact_force1.grad[None].fill(0.0)
        self.predict_force1.grad[None].fill(0.0)
        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.p_sensor1.grad.fill(0.0)
        self.o_sensor1.grad.fill(0.0)

    def clear_traj_grad(self):
        self.fem_sensor1.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.fem_sensor1.clear_step_grad(self.num_sub_frames)
        self.mpm_object.clear_step_grad(self.num_sub_frames)

    def reset(self):
        self.fem_sensor1.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0
        self.contact_force1[None].fill(0.0)
        self.predict_force1[None].fill(0.0)

    @ti.kernel
    def clamp_grid(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid
        ):
            self.mpm_object.grid_m.grad[f, i, j, k] = ti.math.clamp(
                self.mpm_object.grid_m.grad[f, i, j, k], -1000.0, 1000.0
            )
        for i in range(self.fem_sensor1.n_verts):
            self.fem_sensor1.pos.grad[f, i] = ti.math.clamp(
                self.fem_sensor1.pos.grad[f, i], -1000.0, 1000.0
            )
            self.fem_sensor1.vel.grad[f, i] = ti.math.clamp(
                self.fem_sensor1.vel.grad[f, i], -1000.0, 1000.0
            )

    @ti.kernel
    def compute_contact_force(self, f: ti.i32):
        for i in range(self.fem_sensor1.num_triangles):
            a, b, c = self.fem_sensor1.contact_seg[i]
            self.contact_force1[None] += (
                1 / 6 * self.fem_sensor1.external_force_field[f, a]
            )
            self.contact_force1[None] += (
                1 / 6 * self.fem_sensor1.external_force_field[f, b]
            )
            self.contact_force1[None] += (
                1 / 6 * self.fem_sensor1.external_force_field[f, c]
            )

    @ti.func
    def calculate_contact_force(self, sdf, norm_v, relative_v):
        shear_factor_p0 = ti.Vector([0.0, 0.0, 0.0])
        shear_vel_p0 = ti.Vector([0.0, 0.0, 0.0])
        relative_vel_p0 = relative_v
        normal_vel_p0 = ti.max(norm_v.dot(relative_vel_p0), 0)
        normal_factor_p0 = (
            -(self.kn[None] + self.kd[None] * normal_vel_p0) * sdf * norm_v
        )
        shear_vel_p0 = relative_vel_p0 - norm_v.dot(relative_vel_p0) * norm_v
        shear_vel_norm_p0 = shear_vel_p0.norm(self.norm_eps)
        if shear_vel_norm_p0 > 1e-4:
            shear_factor_p0 = (
                1.0
                * (shear_vel_p0 / shear_vel_norm_p0)
                * ti.min(
                    self.kt[None] * shear_vel_norm_p0,
                    self.friction_coeff[None] * normal_factor_p0.norm(self.norm_eps),
                )
            )
        ext_v = normal_factor_p0 + shear_factor_p0
        return ext_v, normal_factor_p0, shear_factor_p0

    @ti.kernel
    def check_collision(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid
        ):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.mpm_object.dx_0,
                        (j + 0.5) * self.mpm_object.dx_0,
                        (k + 0.5) * self.mpm_object.dx_0,
                    ]
                )
                min_idx1 = self.fem_sensor1.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k] = min_idx1

    @ti.kernel
    def collision(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid
        ):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.mpm_object.dx_0,
                        (j + 0.5) * self.mpm_object.dx_0,
                        (k + 0.5) * self.mpm_object.dx_0,
                    ]
                )
                cur_v = self.mpm_object.grid_v_in[f, i, j, k] / (
                    self.mpm_object.grid_m[f, i, j, k] + self.mpm_object.eps
                )
                min_idx1 = self.contact_idx[f, i, j, k]
                cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = (
                    self.fem_sensor1.find_sdf(cur_p, cur_v, min_idx1, f)
                )
                if contact_flag1:
                    ext_v1, _, _ = self.calculate_contact_force(
                        cur_sdf1, -1 * cur_norm_v1, -1 * cur_relative_v1
                    )
                    self.mpm_object.update_contact_force(ext_v1, f, i, j, k)
                    self.fem_sensor1.update_contact_force(min_idx1, -1 * ext_v1, f)

    def memory_to_cache(self, t):
        self.fem_sensor1.memory_to_cache(t)
        self.mpm_object.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.fem_sensor1.memory_from_cache(t)
        self.mpm_object.memory_from_cache(t)

    @ti.kernel
    def compute_angle(self, t: ti.i32):
        for f in range(self.num_sub_frames - 1):
            if f == 0:
                self.angle[t + 1] = self.angle[t] + ti.atan2(
                    self.mpm_object.R[f][1, 0], self.mpm_object.R[f][1, 1]
                )
            else:
                self.angle[t + 1] += ti.atan2(
                    self.mpm_object.R[f][1, 0], self.mpm_object.R[f][1, 1]
                )
    
    def set_up_target_marker_positions(self):
        """
        Reorders the experimental marker positions to match the simulation marker indexing convention.
        Uses the Hungarian algorithm to find optimal marker-to-marker mapping based on squared Euclidean distances
        between markers in the base frame (frame 0).
        """
        # Load target marker positions from pickle file
        with open('difftactile/sensor_model/markers-paired.pkl', 'rb') as f:
            marker_data = pickle.load(f)
        # Create taichi field with appropriate shape
        self.target_marker_positions = ti.Vector.field(marker_data.shape[2], dtype=ti.f32, shape=(marker_data.shape[0], marker_data.shape[1]))
        # Copy data from numpy array to taichi field
        self.target_marker_positions.from_numpy(marker_data)

        # Get base frame markers from experiment and simulation
        exp_base_markers = self.target_marker_positions[0]  # Shape: (num_markers, 2)
        sim_base_markers = self.fem_sensor1.virtual_markers.to_numpy()  # Shape: (num_markers, 2)

        # Compute cost matrix (squared Euclidean distances)
        cost_matrix = cdist(exp_base_markers, sim_base_markers, metric='sqeuclidean')

        # Use Hungarian algorithm to find optimal assignment minimizing sum of squared distances
        exp_indices, sim_indices = linear_sum_assignment(cost_matrix)

        # Create a mapping from experimental indices to simulation indices
        index_mapping = {exp_idx: sim_idx for exp_idx, sim_idx in zip(exp_indices, sim_indices)}

        # Create a new array with reordered experimental markers
        num_frames = self.target_marker_positions.shape[0]
        num_markers = self.target_marker_positions.shape[1]
        reordered_markers = np.zeros_like(self.target_marker_positions)

        # For each frame, reorder the markers according to the mapping
        for frame_idx in range(num_frames):
            for exp_idx, sim_idx in index_mapping.items():
                reordered_markers[frame_idx, sim_idx] = self.target_marker_positions[frame_idx, exp_idx]

        # Update target_marker_positions with reordered markers
        self.target_marker_positions = reordered_markers 

    @ti.kernel
    def compute_marker_loss(self, frame_index: ti.i32):
        """
        Compute RMSE loss between experimental and simulated marker positions for a given frame.
        
        Args:
            frame_index: Index of the frame to compute loss for
        """
        squared_error_sum = 0.0
        
        # Iterate through all markers and accumulate squared errors
        for i in range(self.fem_sensor1.num_markers):
            # Get experimental and simulated marker positions
            exp_marker = self.target_marker_positions[frame_index, i]
            sim_marker = self.fem_sensor1.virtual_markers[i]
            
            # Compute squared error for this marker pair
            dx = exp_marker[0] - sim_marker[0]
            dy = exp_marker[1] - sim_marker[1]
            squared_error = dx * dx + dy * dy
            squared_error_sum += squared_error
            
        # Compute RMSE and add to total loss
        rmse = ti.sqrt(squared_error_sum / self.fem_sensor1.num_markers)
        self.loss[None] += rmse


def main():
    ti.init(debug=False, offline_cache=False, arch=ti.gpu, device_memory_GB=9)

    gui_tuple = set_up_gui()

    phantom_name = "J03_2.obj"
    num_sub_frames = 50
    num_frames = 150
    num_opt_steps = 100
    dt = 5e-5
    contact_model = Contact(
        use_tactile=USE_TACTILE,
        use_state=USE_STATE,
        dt=dt,
        num_frames=num_frames,
        num_sub_frames=num_sub_frames,
        obj=phantom_name,
    )
    losses = []
    contact_model.draw_table()
    contact_model.init_pos_control()
    form_loss = 0
    for opts in range(num_opt_steps):
        print("Opt # step ======================", opts)
        contact_model.init()
        contact_model.clear_all_grad()
        contact_model.clear_state_loss_grad()
        for ts in range(num_frames - 1):
            contact_model.set_pos_control(ts)
            contact_model.fem_sensor1.set_pose_control()
            contact_model.fem_sensor1.set_control_vel(0)
            contact_model.fem_sensor1.set_vel(0)
            contact_model.reset()
            for ss in range(num_sub_frames - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)
            print("# FP Iter ", ts)
            contact_model.compute_contact_force(num_sub_frames - 2)
            form_loss = contact_model.loss[None]
            contact_model.compute_marker_loss(ts)
            print("contact force: ", contact_model.predict_force1[None])
            print("marker loss", contact_model.loss[None] - form_loss)

            update_gui(contact_model, gui_tuple)

        loss_trajectory = 0
        for ts in range(num_frames - 2, -1, -1):
            print("BP", ts)
            contact_model.clear_all_grad()
            contact_model.compute_marker_loss.grad(ts)
            contact_model.compute_contact_force.grad(num_sub_frames - 2)
            for ss in range(num_sub_frames - 2, -1, -1):
                contact_model.update_grad(ss)
            contact_model.fem_sensor1.set_vel.grad(0)
            contact_model.fem_sensor1.set_control_vel.grad(0)
            contact_model.fem_sensor1.set_pose_control.grad()
            contact_model.set_pos_control.grad(ts)

            grad_friction_coeff = contact_model.friction_coeff.grad[ts]
            grad_kn = contact_model.kn.grad[ts]
            grad_kd = contact_model.kd.grad[ts]
            grad_kt = contact_model.kt.grad[ts]
            grad_mu = contact_model.fem_sensor1.mu.grad[ts]
            grad_lam = contact_model.fem_sensor1.lam.grad[ts]

            lr_friction_coeff = 1e-3
            lr_kn = 1e-3
            lr_kd = 1e-3
            lr_kt = 1e-3
            lr_mu = 1e-3
            lr_lam = 1e-3

            contact_model.friction_coeff[ts] -= lr_friction_coeff * grad_friction_coeff
            contact_model.kn[ts] -= lr_kn * grad_kn
            contact_model.kd[ts] -= lr_kd * grad_kd
            contact_model.kt[ts] -= lr_kt * grad_kt
            contact_model.fem_sensor1.mu[ts] -= lr_mu * grad_mu
            contact_model.fem_sensor1.lam[ts] -= lr_lam * grad_lam

            loss_trajectory += contact_model.loss[None]
            print("# BP Iter: ", ts, " loss: ", contact_model.loss[None])
            print("P/O grads: ", grad_friction_coeff, grad_kn, grad_kd, grad_kt, grad_mu, grad_lam)
            print(
                "P/O updated: ",
                contact_model.friction_coeff[ts],
                contact_model.kn[ts],
                contact_model.kd[ts],
                contact_model.kt[ts],
                contact_model.fem_sensor1.mu[ts],
                contact_model.fem_sensor1.lam[ts],
            )
            if (ts - 1) >= 0:
                contact_model.memory_from_cache(ts - 1)
                contact_model.set_pos_control(ts - 1)
                contact_model.fem_sensor1.set_pose_control_bp()
                contact_model.fem_sensor1.set_control_vel(0)
                contact_model.fem_sensor1.set_vel(0)
                contact_model.reset()
                for ss in range(num_sub_frames - 1):
                    contact_model.update(ss)
        losses.append(loss_trajectory)
        if loss_trajectory <= np.min(losses):
            best_friction_coeff = contact_model.friction_coeff.to_numpy()
            best_kn = contact_model.kn.to_numpy()
            best_kd = contact_model.kd.to_numpy()
            best_kt = contact_model.kt.to_numpy()
            best_mu = contact_model.fem_sensor1.mu.to_numpy()
            best_lam = contact_model.fem_sensor1.lam.to_numpy()
    print('best_friction_coeff', best_friction_coeff)
    print('best_kn', best_kn)
    print('best_kd', best_kd)
    print('best_kt', best_kt)
    print('best_mu', best_mu)
    print('best_lam', best_lam)


if __name__ == "__main__":
    main()
