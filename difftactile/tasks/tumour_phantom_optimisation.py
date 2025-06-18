from difftactile.tasks.tumour_phantom_visualisation import (
    ContactVisualisation,
    set_up_gui,
    update_gui,
)
from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.multi_obj import MultiObj

import taichi as ti
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import sys

RUN_ON_LAB_MACHINE = True
SLOW_DOWN = 0.5
SPEED_1_MM_S = 4.0828765820486765 / SLOW_DOWN
SPEED_2_DEG_S = 81.63 / SLOW_DOWN
TIME_STEPS_PER_S = 10 * SLOW_DOWN

@ti.data_oriented
class Contact(ContactVisualisation):
    def __init__(
        self,
        dt,
        num_frames,
        num_sub_frames,
        obj,
    ):
        super().__init__()

        self.dt = dt
        self.num_frames = num_frames
        self.num_sub_frames = num_sub_frames
        self.fem_sensor1 = FEMDomeSensor(dt, num_sub_frames)
        self.mpm_object = MultiObj(
            dt=dt,
            sub_steps=num_sub_frames,
            obj_name=obj,
            space_scale=30.0,
            obj_scale=15.0,
            density=1.5,
            rho=1.07,
        )
        self.set_up_initial_positions()

        # Initialize keypoint indices
        self.keypoint_indices = self.fem_sensor1.get_keypoint_indices(0)

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

        

        self.num_sensor = 1
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
        self.dim = 3
        self.p_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=True
        )
        self.o_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=True
        )
        self.loss = ti.field(float, (), needs_grad=True)
        self.contact_detect_flag = ti.field(float, (), needs_grad=True)
        self.norm_eps = 1e-11
        self.squared_error_sum = ti.field(dtype=float, shape=(), needs_grad=True)
        self.squared_error_sum[None] = 0

        self.set_up_target_marker_positions()
        self.init_visualisation()

        self.trajectory_frame_ix = ti.field(dtype=float, shape=(), needs_grad=False)
        self.trajectory_frame_ix[None] = 0

        self.skip_frames = ti.field(dtype=float, shape=(), needs_grad=False)
        self.skip_frames[None] = 0

        self.velocities_npy = np.array([
            [0, SPEED_1_MM_S, 0, 0, 0, 0],
            [SPEED_1_MM_S, 0, 0, 0, 0, 0],
        ], dtype=float)
        self.time_durations_s_npy = np.array([
            10,
            10,
        ], dtype=float)
        self.velocities = ti.Vector.field(6, dtype=float, shape=self.velocities_npy.shape[0], needs_grad=False)
        self.time_durations_s = ti.field(dtype=float, shape=self.time_durations_s_npy.shape[0], needs_grad=False)
        self.velocities.from_numpy(self.velocities_npy)
        self.time_durations_s.from_numpy(self.time_durations_s_npy)

        self.fill_out_motion_start_end_ixs()

        self.interpolation_exp_frame_start = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.interpolation_exp_frame_end = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.interpolation_alpha = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

        v_npy = np.array([
            [10, 0, 0, 0, 0, 0],
            [-10, 0, 0, 0, 0, 0],

            [0, 10, 0, 0, 0, 0],
            [0, -10, 0, 0, 0, 0],
            
            [0, 0, 10, 0, 0, 0],
            [0, 0, -10, 0, 0, 0],
            
            [0, 0, 0, 90, 0, 0],
            [0, 0, 0, -90, 0, 0],
            
            [0, 0, 0, 0, 90, 0],
            [0, 0, 0, 0, -90, 0],
            
            [0, 0, 0, 0, 0, 90],
            [0, 0, 0, 0, 0, -90]
        ], dtype=float)
        self.v = ti.Vector.field(6, dtype=float, shape=v_npy.shape[0], needs_grad=False)
        self.v.from_numpy(v_npy)

        # PID controller parameters
        self.pid_controller_kp = ti.field(dtype=float, shape=(), needs_grad=True)  # Proportional gain
        self.pid_controller_ki = ti.field(dtype=float, shape=(), needs_grad=True)  # Integral gain
        self.pid_controller_kd = ti.field(dtype=float, shape=(), needs_grad=True)  # Derivative gain
        self.pid_controller_kp[None] = 10.0  # Initial values - these may need tuning
        self.pid_controller_ki[None] = 0.0
        self.pid_controller_kd[None] = 0.0
        
        # Error accumulation for integral term
        self.pos_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)
        self.ori_error_sum = ti.Vector.field(4, dtype=float, shape=(), needs_grad=True)
        
        # Previous error for derivative term
        self.prev_pos_error = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)
        self.prev_ori_error = ti.Vector.field(4, dtype=float, shape=(), needs_grad=True)

        target_positions_npy = np.array([
            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5+5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5, 11.5+5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5, 11.5, 3.00625+50+5, -0.7071068, 0, 0, 0.7071068],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5, 11.5, 3.00625+50, 0, 0, 0, 1],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5, 11.5, 3.00625+50, -0.5, 0.5, -0.5, 0.5],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
            [12.5, 11.5, 3.00625+50, -0.5, 0.5, 0.5, 0.5],

            [12.5, 11.5, 3.00625+50, -0.7071068, 0, 0, 0.7071068],
        ], dtype=float)
        self.target_positions = ti.Vector.field(7, dtype=float, shape=target_positions_npy.shape[0], needs_grad=False)
        self.target_positions.from_numpy(target_positions_npy)

        # Add fields to track current target and control state
        self.current_target_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.current_target_idx[None] = 0
        self.position_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.position_tolerance[None] = 0.1  # 1mm tolerance
        self.orientation_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.orientation_tolerance[None] = 1.0  # 1 degree tolerance
        
        # Add fields for dwell time control
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = 50  # Number of frames to stay at each target
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.is_dwelling[None] = False

    def set_up_initial_positions(self):
        phantom_pose = [12.5, 11.5, 2.05625, 0, 0, 0]
        tactile_sensor_pose = [12.5, 11.5, 6.30625+50, -90, 0, 0]
        
        self.mpm_object.init(
            pos=phantom_pose[:3],
            ori=phantom_pose[3:],
            vel=[0.0, 0.0, 0.0],
        )
        t_dx, t_dy, t_dz, rot_x, rot_y, rot_z = tactile_sensor_pose
        self.fem_sensor1.init(rot_x, rot_y, rot_z, t_dx, t_dy, t_dz)

        self.tactile_sensor_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.phantom_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.tactile_sensor_initial_position[0] = ti.Vector(tactile_sensor_pose[:3])
        self.phantom_initial_position[0] = ti.Vector(phantom_pose[:3])

    def fill_out_motion_start_end_ixs(self):
        self.motion_start_end_ixs = ti.field(dtype=int, shape=self.velocities_npy.shape[0] + 1, needs_grad=False)
        i = self.skip_frames[None]
        res = [i]
        for j in range(self.time_durations_s_npy.shape[0]):
            i += self.time_durations_s_npy[j] * TIME_STEPS_PER_S
            res.append(i)
        res_npy = np.array(res, dtype=int)
        self.motion_start_end_ixs.from_numpy(res_npy)

        arr = np.array([
            3, 8, 13
        ], dtype=int)
        self.motion_start_end_experimental_video_frame_ixs = ti.field(dtype=int, shape=self.velocities_npy.shape[0] + 1, needs_grad=False)
        self.motion_start_end_experimental_video_frame_ixs.from_numpy(arr)

    @ti.kernel
    def set_up_trajectory(self):
        for i in range(self.num_frames):
            self.p_sensor1[i] = ti.Vector([0, 0, 0], dt=ti.f32)
            self.o_sensor1[i] = ti.Vector([0, 0, 0], dt=ti.f32)
        
        frames_to_skip = 0
        frames_per_motion = 100
        
        for i in range(self.v.shape[0]):
            for j in range(frames_to_skip + i * frames_per_motion, frames_to_skip + (i+1) * frames_per_motion):
                self.p_sensor1[j] = ti.Vector([self.v[i][0], self.v[i][1], self.v[i][2]], dt=ti.f32)
                self.o_sensor1[j] = ti.Vector([self.v[i][3], self.v[i][4], self.v[i][5]], dt=ti.f32)

    def set_pos_control_maybe_print(self, f: int):
        if False:
            print("\nInput position vector (p_sensor1):")
            print(self.p_sensor1[f].to_numpy())
            print("\nInput orientation vector (o_sensor1):")
            print(self.o_sensor1[f].to_numpy())
            print("\nSet position vector (d_pos):")
            print(self.fem_sensor1.d_pos_global[None].to_numpy())
            print("\nSet orientation vector (d_ori):")
            print(self.fem_sensor1.d_ori_global_euler_angles[None].to_numpy())
            print()

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
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0
        self.contact_detect_flag.grad[None] = 0.0
        self.loss[None] = 0.0
        self.loss.grad[None] = 0.0
        self.p_sensor1.grad.fill(0.0)
        self.o_sensor1.grad.fill(0.0)
        self.squared_error_sum.grad[None] = 0.0

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
        self.squared_error_sum[None] = 0.0

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
    
    def set_up_target_marker_positions(self):
        """
        Reorders the experimental marker positions to match the simulation marker indexing convention.
        Uses the Hungarian algorithm to find optimal marker-to-marker mapping based on squared Euclidean distances
        between markers in the base frame (frame 0).
        """
        with open(f'../sensor_model/markers-paired.pkl', 'rb') as f:
            marker_data = pickle.load(f)
        self.experiment_num_frames = marker_data.shape[0]
        self.experiment_num_markers = marker_data.shape[1]
        cost_matrix = cdist(marker_data[0], self.fem_sensor1.virtual_markers.to_numpy(), metric='sqeuclidean')
        exp_indices, sim_indices = linear_sum_assignment(cost_matrix)
        index_mapping = {exp_idx: sim_idx for exp_idx, sim_idx in zip(exp_indices, sim_indices)}
        reordered_markers = np.zeros_like(marker_data)
        for frame_idx in range(self.experiment_num_frames):
            for exp_idx, sim_idx in index_mapping.items():
                reordered_markers[frame_idx, sim_idx] = marker_data[frame_idx, exp_idx]

        self.target_marker_positions = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.experiment_num_frames, self.experiment_num_markers), needs_grad=True
        )
        self.target_marker_positions.from_numpy(reordered_markers)

    @ti.kernel
    def interpolate_experimental_video(self, f: ti.i32):
        # Find which motion segment this frame falls into
        motion_segment = -1
        for i in range(self.motion_start_end_ixs.shape[0] - 1):
            if f >= self.motion_start_end_ixs[i] and f < self.motion_start_end_ixs[i + 1] and motion_segment == -1:
                motion_segment = i
        
        # Get the experimental video frame indices for this motion segment
        self.interpolation_exp_frame_start[None] = self.motion_start_end_experimental_video_frame_ixs[motion_segment]
        self.interpolation_exp_frame_end[None] = self.motion_start_end_experimental_video_frame_ixs[motion_segment + 1]
        
        # Calculate interpolation factor based on relative position in motion segment
        sim_segment_start = self.motion_start_end_ixs[motion_segment]
        sim_segment_end = self.motion_start_end_ixs[motion_segment + 1]
        self.interpolation_alpha[None] = (f - sim_segment_start) / (sim_segment_end - sim_segment_start)

    @ti.kernel
    def compute_marker_loss_1(self, f: ti.i32):
        """
        Compute RMSE loss between experimental and simulated marker positions for a given frame.
        Uses linear interpolation between experimental video frames based on motion segments.
        
        Args:
            f: Index of the frame to compute loss for
        """
        # Iterate through all markers and accumulate squared errors
        for i in range(self.fem_sensor1.num_markers):
            # Get experimental marker positions at start and end of segment
            exp_marker_start = self.target_marker_positions[self.interpolation_exp_frame_start[None], i]
            exp_marker_end = self.target_marker_positions[self.interpolation_exp_frame_end[None], i]
            
            # Interpolate experimental marker position
            exp_marker = exp_marker_start * (1 - self.interpolation_alpha[None]) + exp_marker_end * self.interpolation_alpha[None]
            
            # Get simulated marker position
            sim_marker = self.fem_sensor1.predict_markers[i]
            
            # Compute squared error for this marker pair
            dx = exp_marker[0] - sim_marker[0]
            dy = exp_marker[1] - sim_marker[1]
            squared_error = dx * dx + dy * dy
            self.squared_error_sum[None] += squared_error

    @ti.kernel
    def compute_marker_loss_2(self):
        # Compute RMSE and add to total loss
        rmse = ti.sqrt(self.squared_error_sum[None] / self.fem_sensor1.num_markers)
        self.loss[None] += rmse

    @ti.kernel
    def compute_pid_control(self):
        # Get current position and orientation using reference keypoint
        current_pos = self.fem_sensor1.pos[0, self.keypoint_indices[0]]
        current_ori = self.fem_sensor1.get_quaternion()
        
        # Get current target position and orientation
        target = self.target_positions[self.current_target_idx[None]]
        target_pos = ti.Vector([target[0], target[1], target[2]])
        target_ori = ti.Vector([target[3], target[4], target[5], target[6]])
        
        # Compute position and orientation errors
        pos_error = target_pos - current_pos
        ori_error = target_ori - current_ori
        
        # Check if current target is reached
        pos_error_magnitude = pos_error.norm()
        ori_error_magnitude = ori_error.norm()
        
        # If target is reached and not already dwelling, start dwelling
        if not self.is_dwelling[None] and pos_error_magnitude < self.position_tolerance[None] and ori_error_magnitude < self.orientation_tolerance[None]:
            self.is_dwelling[None] = True
            self.dwell_counter[None] = 0
        
        # If dwelling, increment counter and check if dwell time is complete
        if self.is_dwelling[None]:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = False
                if self.current_target_idx[None] < self.target_positions.shape[0] - 1:
                    self.current_target_idx[None] += 1
                    # Reset error sums when switching targets
                    self.pos_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.ori_error_sum[None] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    self.prev_pos_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_ori_error[None] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    
                    # Get new target position and orientation
                    target = self.target_positions[self.current_target_idx[None]]
                    target_pos = ti.Vector([target[0], target[1], target[2]])
                    target_ori = ti.Vector([target[3], target[4], target[5], target[6]])
                    
                    # Recompute errors for new target
                    pos_error = target_pos - current_pos
                    ori_error = target_ori - current_ori
        
        # If dwelling, set control outputs to zero to maintain position
        if self.is_dwelling[None]:
            self.fem_sensor1.d_pos_global[None] = ti.Vector([0.0, 0.0, 0.0])
            self.fem_sensor1.d_ori_global_quaternion[None] = ti.Vector([0.0, 0.0, 0.0, 1.0])

            if True:
                print(f'We are dwelling')
                print()
        else:
            # Update error sums for integral term
            self.pos_error_sum[None] += pos_error * self.dt
            self.ori_error_sum[None] += ori_error * self.dt
            
            # Compute derivative term
            pos_derivative = (pos_error - self.prev_pos_error[None]) / self.dt
            ori_derivative = (ori_error - self.prev_ori_error[None]) / self.dt
            
            # Store current error for next iteration
            self.prev_pos_error[None] = pos_error
            self.prev_ori_error[None] = ori_error
            
            # Compute PID control output
            pos_control = self.pid_controller_kp[None] * pos_error + self.pid_controller_ki[None] * self.pos_error_sum[None] + self.pid_controller_kd[None] * pos_derivative
            ori_control = self.pid_controller_kp[None] * ori_error + self.pid_controller_ki[None] * self.ori_error_sum[None] + self.pid_controller_kd[None] * ori_derivative

            # Set control outputs
            self.fem_sensor1.d_pos_global[None] = pos_control
            self.fem_sensor1.d_ori_global_quaternion[None] = ori_control
        
            if False:
                # Print all variables used in the function
                print("\nPID Control Variables:")
                print(f"Current Position (current_pos): {current_pos}")
                print(f"Current Orientation (current_ori): {current_ori}")
                print(f"Target Position (target_pos): {target_pos}")
                print(f"Target Orientation (target_ori): {target_ori}")
                print(f"Position Error (pos_error): {pos_error}")
                print(f"Orientation Error (ori_error): {ori_error}")
                print(f"Position Error Magnitude (pos_error_magnitude): {pos_error_magnitude}")
                print(f"Orientation Error Magnitude (ori_error_magnitude): {ori_error_magnitude}")
                print(f"Position Error Sum (self.pos_error_sum[None]): {self.pos_error_sum[None]}")
                print(f"Orientation Error Sum (self.ori_error_sum[None]): {self.ori_error_sum[None]}")
                print(f"Previous Position Error (self.prev_pos_error[None]): {self.prev_pos_error[None]}")
                print(f"Previous Orientation Error (self.prev_ori_error[None]): {self.prev_ori_error[None]}")
                print(f"Position Derivative (pos_derivative): {pos_derivative}")
                print(f"Orientation Derivative (ori_derivative): {ori_derivative}")
                print(f"Position Control Output (pos_control): {pos_control}")
                print(f"Orientation Control Output (ori_control): {ori_control}")
                print(f"Current Target Index (self.current_target_idx[None]): {self.current_target_idx[None]}")
                print(f"Is Dwelling (self.is_dwelling[None]): {self.is_dwelling[None]}")
                print(f"Dwell Counter (self.dwell_counter[None]): {self.dwell_counter[None]}")
                print()

def main():
    np.set_printoptions(precision=3, floatmode='maxprec', suppress=False)
    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cuda, device_memory_GB=9)
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    gui_tuple = set_up_gui()

    phantom_name = "suturing-phantom.stl"
    num_sub_frames = 50
    num_frames = 2_000
    num_opt_steps = 20
    dt = 5e-5
    contact_model = Contact(
        dt=dt,
        num_frames=num_frames,
        num_sub_frames=num_sub_frames,
        obj=phantom_name,
    )
    contact_model.set_up_trajectory()
    np.savetxt(f'output/trajectory.p_sensor1.csv', contact_model.p_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    np.savetxt(f'output/trajectory.o_sensor1.csv', contact_model.o_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    
    for opts in range(num_opt_steps):
        print(f"optimisation step: {opts}")
        contact_model.set_up_initial_positions()
        contact_model.clear_all_grad()
        print('forward')
        for ts in range(num_frames - 1):
            print(f'forward time step: {ts}')
            contact_model.compute_pid_control()
            contact_model.fem_sensor1.set_pose_control()
            contact_model.fem_sensor1.set_pose_control_maybe_print()
            contact_model.reset()
            for ss in range(num_sub_frames - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)
            contact_model.interpolate_experimental_video(ts)
            contact_model.compute_marker_loss_1(ts)
            contact_model.compute_marker_loss_2()
            
            keypoint_coords = contact_model.fem_sensor1.get_keypoint_coordinates(0, contact_model.keypoint_indices)
            keypoint_coords = np.vstack([keypoint_coords, np.array([12.5+5, 11.5, 6.30625+50])])
            update_gui(contact_model, gui_tuple, num_frames, ts, keypoint_coords)

            # if ts == 200:
            #     sys.exit()

            if ts in set([0, 10, 100]):
                np.savetxt(f'output/tactile_sensor.pos.time_step.{ts}.csv', contact_model.fem_sensor1.pos.to_numpy()[0], delimiter=",", fmt='%.2f')
            
        contact_model.loss.grad[None] = 1.0
        
        print('backward')
        for ts in range(num_frames - 2, -1, -1):
            print(f'backward time step: {ts}')
            contact_model.compute_marker_loss_2.grad()
            contact_model.compute_marker_loss_1.grad(ts)
            for ss in range(num_sub_frames - 2, -1, -1):
                contact_model.update_grad(ss)
            contact_model.fem_sensor1.set_pose_control.grad()

            if (ts - 1) >= 0:
                contact_model.memory_from_cache(ts - 1)
                contact_model.set_pos_control(ts - 1)
                contact_model.fem_sensor1.set_pose_control_bp()
                contact_model.reset()
                for ss in range(num_sub_frames - 1):
                    contact_model.update(ss)
            
            keypoint_coords = contact_model.fem_sensor1.get_keypoint_coordinates(0, contact_model.keypoint_indices)
            keypoint_coords = np.vstack([keypoint_coords, np.array([12.5+5, 11.5, 6.30625+50])])
            update_gui(contact_model, gui_tuple, num_frames, ts, keypoint_coords)
        
        print(f"Accumulated gradients after optimisation step {opts}:")
        print(f"kn grad: {contact_model.kn.grad[None]}")
        print(f"kd grad: {contact_model.kd.grad[None]}")
        print(f"kt grad: {contact_model.kt.grad[None]}")
        print(f"friction_coeff grad: {contact_model.friction_coeff.grad[None]}")
        print(f"mu grad: {contact_model.fem_sensor1.mu.grad[None]}")
        print(f"lam grad: {contact_model.fem_sensor1.lam.grad[None]}")
        print()

        contact_model.friction_coeff[None] -= 1e3 * contact_model.friction_coeff.grad[None]
        contact_model.kn[None] -= 1e3 * contact_model.kn.grad[None]
        contact_model.kd[None] -= 1e3 * contact_model.kd.grad[None]
        contact_model.kt[None] -= 1e3 * contact_model.kt.grad[None]
        contact_model.fem_sensor1.mu[None] -= 1e3 * contact_model.fem_sensor1.mu.grad[None]
        contact_model.fem_sensor1.lam[None] -= 1e3 * contact_model.fem_sensor1.lam.grad[None]


if __name__ == "__main__":
    main()
