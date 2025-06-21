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
PHANTOM_INITIAL_POSE = [9.75, 9.75, 1.85, 0, 0, 0]
SENSOR_DOME_TIP_INITIAL_POSE = [9.75, 9.75, 2.95, -90, 0, 0]

def print_point_cloud(arr):
    # Print the shape for verification
    print('Shape:', arr.shape)

    # Print min and max along axis 1
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    print('Min along axis 0:', min_vals)
    print('Max along axis 0:', max_vals)
    print('diff', max_vals - min_vals)

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
        self.tactile_sensor = FEMDomeSensor(dt, num_sub_frames)
        self.phantom = MultiObj(
            dt=dt,
            sub_steps=num_sub_frames,
            obj_name=obj,
            space_scale=16.0,
            obj_scale=8.0,
            density=0.5,
            rho=1.07,
        )

        self.tactile_sensor_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.phantom_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.trajectory = ti.Vector.field(6, dtype=float, shape=3, needs_grad=False)
        self.set_up_initial_positions_and_trajectory()

        # Initialize keypoint indices
        self.keypoint_indices = self.tactile_sensor.get_keypoint_indices(0)

        self.kn = ti.field(dtype=float, shape=(), needs_grad=False)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=False)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=False)
        self.kn[None] = 34.53
        self.kd[None] = 269.44
        self.kt[None] = 108.72
        self.friction_coeff[None] = 14.16
        self.tactile_sensor.mu[None] = 1294.01
        self.tactile_sensor.lam[None] = 9201.11

        self.num_sensor = 1
        self.contact_idx = ti.Vector.field(
            self.num_sensor,
            dtype=int,
            shape=(
                self.num_sub_frames,
                self.phantom.n_grid,
                self.phantom.n_grid,
                self.phantom.n_grid,
            ),
        )
        self.dim = 3
        self.p_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=False
        )
        self.o_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.num_frames), needs_grad=False
        )
        self.loss = ti.field(float, (), needs_grad=False)
        self.contact_detect_flag = ti.field(float, (), needs_grad=False)
        self.norm_eps = 1e-11
        self.squared_error_sum = ti.field(dtype=float, shape=(), needs_grad=False)
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

        # PID controller parameters
        self.pid_controller_kp = ti.field(dtype=float, shape=(), needs_grad=False)  # Proportional gain
        self.pid_controller_ki = ti.field(dtype=float, shape=(), needs_grad=False)  # Integral gain
        self.pid_controller_kd = ti.field(dtype=float, shape=(), needs_grad=False)  # Derivative gain
        self.pid_controller_kp[None] = 100.0  # Initial values - these may need tuning
        self.pid_controller_ki[None] = 0.0
        self.pid_controller_kd[None] = 0.0
        
        # Error accumulation for integral term
        self.pos_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.ori_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)  # Changed from 4 to 3 for Euler angles
        
        # Previous error for derivative term
        self.prev_pos_error = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.prev_ori_error = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)  # Changed from 4 to 3 for Euler angles

        # Add fields to track current target and control state
        self.current_target_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.current_target_idx[None] = 0
        self.position_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.position_tolerance[None] = 0.1  # 1 mm tolerance
        self.orientation_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.orientation_tolerance[None] = 1  # 1 degree tolerance
        
        # Add fields for dwell time control
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = 0 # Number of frames to stay at each target
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.is_dwelling[None] = False
        self.last_target_reached = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.last_target_reached[None] = False

    def set_up_initial_positions_and_trajectory(self):
        ix = self.tactile_sensor.get_keypoint_indices_numpy_point_a()
        camera_lens_to_sensor_tip = self.tactile_sensor.all_nodes[ix, 1]
        self.phantom_pose = PHANTOM_INITIAL_POSE.copy()
        
        # Draw random cylinder parameters
        cx = np.random.uniform(-1.0, 1.0)
        cy = np.random.uniform(-1.0, 1.0)
        cz = np.random.uniform(-0.9, 0.9)
        theta = np.random.uniform(0, 90)
        h = np.random.uniform(1, 6)
        r = np.random.uniform(0.1, 0.4)
        # cylinder_tuple = (cx, cy, cz, theta, h, r)
        cylinder_tuple = (0.0, 0.0, 0.6, 0.0, 6.0, 0.4)
        stiffness_healthy_tissue = np.random.uniform(2.5e3, 7.5e3)
        stiffness_tumour = np.random.uniform(2.5e4, 7.5e4)
        # stiffness_tuple = (stiffness_healthy_tissue, stiffness_tumour)
        stiffness_tuple = (5e3, 5e4)
        # Draw tumour_present: True with 80%, False with 20%
        # tumour_present = np.random.rand() < 0.8
        tumour_present = True

        self.phantom.init(
            pos=self.phantom_pose[:3],
            ori=self.phantom_pose[3:],
            vel=[0.0, 0.0, 0.0],
            cylinder_tuple=cylinder_tuple,
            stiffness_tuple=stiffness_tuple,
            tumour_present=tumour_present,
        )

        xr_offset = np.random.uniform(-15, 15)
        press_depth = np.random.uniform(0.6, 1.6)
        x, y, z, xr, yr, zr = SENSOR_DOME_TIP_INITIAL_POSE
        trajectory_npy = np.array([
            [x, y, z, xr, yr, zr],
            [x, y, z-1.6, xr, yr, zr],
        ], dtype=float)
        self.trajectory.from_numpy(trajectory_npy)

        self.sensor_dome_tip_initial_pose = trajectory_npy[0].tolist()
        self.sensor_dome_tip_initial_pose[2] += camera_lens_to_sensor_tip
        t_dx, t_dy, t_dz, rot_x, rot_y, rot_z = self.sensor_dome_tip_initial_pose
        self.tactile_sensor.init(rot_x, rot_y, rot_z, t_dx, t_dy, t_dz)
        self.tactile_sensor_initial_position[0] = ti.Vector(self.sensor_dome_tip_initial_pose[:3])
        self.phantom_initial_position[0] = ti.Vector(self.phantom_pose[:3])
    
        
    def reset_pid_controller(self):
        self.pos_error_sum.fill(0)
        self.ori_error_sum.fill(0)
        self.prev_pos_error.fill(0)
        self.prev_ori_error.fill(0)
        self.current_target_idx[None] = 0
        self.dwell_counter[None] = 0
        self.is_dwelling[None] = False
        self.last_target_reached[None] = False

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

    def set_pos_control_maybe_print(self, f: int):
        if False:
            print("\nInput position vector (p_sensor1):")
            print(self.p_sensor1[f].to_numpy())
            print("\nInput orientation vector (o_sensor1):")
            print(self.o_sensor1[f].to_numpy())
            print("\nSet position vector (d_pos):")
            print(self.tactile_sensor.d_pos_global[None].to_numpy())
            print("\nSet orientation vector (d_ori):")
            print(self.tactile_sensor.d_ori_global_euler_angles[None].to_numpy())
            print()

    def update(self, f):
        self.phantom.compute_new_F(f)
        self.phantom.svd(f)
        self.phantom.p2g(f)
        self.tactile_sensor.update_internal_forces(f)
        self.phantom.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.phantom.grid_op(f)
        self.phantom.g2p(f)
        self.tactile_sensor.update_external_forces(f)

    def update_grad(self, f):
        self.tactile_sensor.update_external_forces.grad(f)
        self.phantom.g2p.grad(f)
        self.phantom.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.tactile_sensor.update_internal_forces.grad(f)
        self.phantom.p2g.grad(f)
        self.phantom.svd_grad(f)
        self.phantom.compute_new_F.grad(f)

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
        self.tactile_sensor.clear_loss_grad()
        self.phantom.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.tactile_sensor.clear_step_grad(self.num_sub_frames)
        self.phantom.clear_step_grad(self.num_sub_frames)

    def reset(self):
        self.tactile_sensor.reset_contact()
        self.phantom.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0
        self.squared_error_sum[None] = 0.0

    @ti.kernel
    def clamp_grid(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            self.phantom.grid_m.grad[f, i, j, k] = ti.math.clamp(
                self.phantom.grid_m.grad[f, i, j, k], -1000.0, 1000.0
            )
        for i in range(self.tactile_sensor.n_verts):
            self.tactile_sensor.pos.grad[f, i] = ti.math.clamp(
                self.tactile_sensor.pos.grad[f, i], -1000.0, 1000.0
            )
            self.tactile_sensor.vel.grad[f, i] = ti.math.clamp(
                self.tactile_sensor.vel.grad[f, i], -1000.0, 1000.0
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
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            if self.phantom.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.phantom.dx_0,
                        (j + 0.5) * self.phantom.dx_0,
                        (k + 0.5) * self.phantom.dx_0,
                    ]
                )
                min_idx1 = self.tactile_sensor.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k] = min_idx1

    @ti.kernel
    def collision(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            if self.phantom.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.phantom.dx_0,
                        (j + 0.5) * self.phantom.dx_0,
                        (k + 0.5) * self.phantom.dx_0,
                    ]
                )
                cur_v = self.phantom.grid_v_in[f, i, j, k] / (
                    self.phantom.grid_m[f, i, j, k] + self.phantom.eps
                )
                min_idx1 = self.contact_idx[f, i, j, k]
                cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = (
                    self.tactile_sensor.find_sdf(cur_p, cur_v, min_idx1, f)
                )
                if contact_flag1:
                    ext_v1, _, _ = self.calculate_contact_force(
                        cur_sdf1, -1 * cur_norm_v1, -1 * cur_relative_v1
                    )
                    self.phantom.update_contact_force(ext_v1, f, i, j, k)
                    self.tactile_sensor.update_contact_force(min_idx1, -1 * ext_v1, f)

    def memory_to_cache(self, t):
        self.tactile_sensor.memory_to_cache(t)
        self.phantom.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.tactile_sensor.memory_from_cache(t)
        self.phantom.memory_from_cache(t)
    
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
        cost_matrix = cdist(marker_data[0], self.tactile_sensor.virtual_markers.to_numpy(), metric='sqeuclidean')
        exp_indices, sim_indices = linear_sum_assignment(cost_matrix)
        index_mapping = {exp_idx: sim_idx for exp_idx, sim_idx in zip(exp_indices, sim_indices)}
        reordered_markers = np.zeros_like(marker_data)
        for frame_idx in range(self.experiment_num_frames):
            for exp_idx, sim_idx in index_mapping.items():
                reordered_markers[frame_idx, sim_idx] = marker_data[frame_idx, exp_idx]

        self.target_marker_positions = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.experiment_num_frames, self.experiment_num_markers), needs_grad=False
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
        for i in range(self.tactile_sensor.num_markers):
            # Get experimental marker positions at start and end of segment
            exp_marker_start = self.target_marker_positions[self.interpolation_exp_frame_start[None], i]
            exp_marker_end = self.target_marker_positions[self.interpolation_exp_frame_end[None], i]
            
            # Interpolate experimental marker position
            exp_marker = exp_marker_start * (1 - self.interpolation_alpha[None]) + exp_marker_end * self.interpolation_alpha[None]
            
            # Get simulated marker position
            sim_marker = self.tactile_sensor.predict_markers[i]
            
            # Compute squared error for this marker pair
            dx = exp_marker[0] - sim_marker[0]
            dy = exp_marker[1] - sim_marker[1]
            squared_error = dx * dx + dy * dy
            self.squared_error_sum[None] += squared_error

    @ti.kernel
    def compute_marker_loss_2(self):
        # Compute RMSE and add to total loss
        rmse = ti.sqrt(self.squared_error_sum[None] / self.tactile_sensor.num_markers)
        self.loss[None] += rmse

    @ti.kernel
    def pid_controller(self, ts: ti.i32):
        # Get current position and orientation using reference keypoint
        current_pos = self.tactile_sensor.pos[0, self.keypoint_indices[0]]
        current_ori = self.tactile_sensor.get_euler_angles()
        
        # Get current target position and orientation
        target = self.trajectory[self.current_target_idx[None]]
        target_pos = ti.Vector([target[0], target[1], target[2]])
        target_ori = ti.Vector([target[3], target[4], target[5]])  # Now using Euler angles
        
        # Compute position and orientation errors
        pos_error = target_pos - current_pos
        ori_error = target_ori - current_ori
        
        # Check if current target is reached
        pos_error_magnitude = pos_error.norm()
        ori_error_magnitude = ori_error.norm()
        
        is_final_target = self.current_target_idx[None] == self.trajectory.shape[0] - 1

        # If target is reached and not already dwelling, start dwelling (only for non-final targets)
        if (not is_final_target and not self.is_dwelling[None] and 
            pos_error_magnitude < self.position_tolerance[None] and 
            ori_error_magnitude < self.orientation_tolerance[None]):
            self.is_dwelling[None] = True
            self.dwell_counter[None] = 0
            if not self.last_target_reached[None]:
                print(f'target {self.current_target_idx[None]} ({target}) reached at time step {ts}!')
        
        # If dwelling, increment counter and check if dwell time is complete
        if self.is_dwelling[None]:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = False
                if self.current_target_idx[None] < self.trajectory.shape[0] - 1:
                    self.current_target_idx[None] += 1
                    # Reset error sums when switching targets
                    self.pos_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.ori_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_pos_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_ori_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    
                    # Get new target position and orientation
                    target = self.trajectory[self.current_target_idx[None]]
                    target_pos = ti.Vector([target[0], target[1], target[2]])
                    target_ori = ti.Vector([target[3], target[4], target[5]])
                    
                    # Recompute errors for new target
                    pos_error = target_pos - current_pos
                    ori_error = target_ori - current_ori
                else:
                    self.last_target_reached[None] = True
        
        # If dwelling, set control outputs to zero to maintain position
        # But if at final target, never dwell, always actively control
        if self.is_dwelling[None]:
            self.tactile_sensor.d_pos_global[None] = ti.Vector([0.0, 0.0, 0.0])
            self.tactile_sensor.d_ori_global_euler_angles[None] = ti.Vector([0.0, 0.0, 0.0])
            if False:
                print(f'We are dwelling')
                print()
        else:
            # Update error sums for integral term
            self.pos_error_sum[None] += pos_error
            self.ori_error_sum[None] += ori_error
            
            # Compute derivative term
            pos_derivative = pos_error - self.prev_pos_error[None]
            ori_derivative = ori_error - self.prev_ori_error[None]
            
            # Store current error for next iteration
            self.prev_pos_error[None] = pos_error
            self.prev_ori_error[None] = ori_error
            
            # Compute PID control output
            pos_control = self.pid_controller_kp[None] * pos_error + self.pid_controller_ki[None] * self.pos_error_sum[None] + self.pid_controller_kd[None] * pos_derivative
            
            clamp_speed = True
            # Clamp pos_control to max_speed
            max_speed_pos = 10.0
            pos_control_norm = pos_control.norm()
            if clamp_speed and pos_control_norm > max_speed_pos:
                pos_control = pos_control / pos_control_norm * max_speed_pos
            
            ori_control = self.pid_controller_kp[None] * ori_error + self.pid_controller_ki[None] * self.ori_error_sum[None] + self.pid_controller_kd[None] * ori_derivative
            
            # Clamp ori_control to max_speed_ori
            max_speed_ori = 10.0
            ori_control_norm = ori_control.norm()
            if clamp_speed and ori_control_norm > max_speed_ori:
                ori_control = ori_control / ori_control_norm * max_speed_ori

            # Set control outputs
            self.tactile_sensor.d_pos_global[None] = pos_control
            self.tactile_sensor.d_ori_global_euler_angles[None] = ori_control
        
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

    gui_tuple = set_up_gui(PHANTOM_INITIAL_POSE.copy(), SENSOR_DOME_TIP_INITIAL_POSE.copy())

    phantom_name = "cylinder.stl"
    num_sub_frames = 50
    num_frames = 5_000
    num_opt_steps = 1
    dt = 5e-5
    contact_model = Contact(
        dt=dt,
        num_frames=num_frames,
        num_sub_frames=num_sub_frames,
        obj=phantom_name,
    )
    np.savetxt(f'output/trajectory.p_sensor1.csv', contact_model.p_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    np.savetxt(f'output/trajectory.o_sensor1.csv', contact_model.o_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    
    for opts in range(num_opt_steps):
        print(f"optimisation step: {opts}")
        contact_model.set_up_initial_positions_and_trajectory()
        contact_model.reset_pid_controller()
        contact_model.reset_3d_scene()
        print('forward')
        for ts in range(num_frames - 1):
            if ts % 50 == 0:
                sensor_mean_deformation_top_10_percent = contact_model.tactile_sensor.compute_mean_deformation_top_10_percent()
                print(f'sensor_mean_deformation_top_10_percent at ts: {ts}: {sensor_mean_deformation_top_10_percent}')
            if ts % 500 == 0:
                pickles = [
                    ('pos', contact_model.tactile_sensor.pos.to_numpy()[0]),
                    ('all_f2v', contact_model.tactile_sensor.all_f2v),
                ]
                for x, y in pickles:
                    with open(f"output/tactile_sensor.ts={ts}.{x}.pkl", 'wb') as f:
                        pickle.dump(y, f)
                    print(f'ts: {ts} pickle dumped!')

            contact_model.pid_controller(ts)
            contact_model.tactile_sensor.set_pose_control()
            contact_model.tactile_sensor.set_pose_control_maybe_print()
            contact_model.tactile_sensor.set_control_vel(0)
            contact_model.tactile_sensor.set_vel(0)
            contact_model.reset()
            for ss in range(num_sub_frames - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(0)            

            keypoint_coords = contact_model.tactile_sensor.get_keypoint_coordinates(0, contact_model.keypoint_indices)
            update_gui(contact_model, gui_tuple, num_frames, ts, keypoint_coords[0, :].reshape((1, 3)))

if __name__ == "__main__":
    main()
