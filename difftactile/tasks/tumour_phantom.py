import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os

off_screen = False
from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.multi_obj import MultiObj
import argparse

enable_gui1 = True
enable_gui2 = False
enable_gui3 = False


@ti.data_oriented
class Contact:
    def __init__(
        self, use_tactile, use_state, dt=5e-5, total_steps=300, sub_steps=50, obj=None
    ):
        self.dt = dt
        self.total_steps = total_steps
        self.sub_steps = sub_steps
        self.fem_sensor1 = FEMDomeSensor(dt, sub_steps)
        self.space_scale = 8.0
        self.obj_scale = 6.0
        self.use_tactile = use_tactile
        self.use_state = use_state
        self.mpm_object = MultiObj(
            dt=dt,
            sub_steps=sub_steps,
            obj_name=obj,
            space_scale=self.space_scale,
            obj_scale=self.obj_scale,
            density=1.5,
            rho=0.2,
        )
        self.alpha = ti.field(float, ())
        self.beta = ti.field(float, ())
        self.alpha[None] = 1e1
        self.beta[None] = 5e-12
        self.num_sensor = 1
        self.init()
        self.view_phi = 0
        self.view_theta = 0
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
                self.sub_steps,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
            ),
        )
        self.dim = 3
        self.p_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True
        )
        self.o_sensor1 = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True
        )
        self.loss = ti.field(float, (), needs_grad=True)
        self.angle = ti.field(float, (self.total_steps), needs_grad=True)
        self.contact_detect_flag = ti.field(float, (), needs_grad=True)
        self.predict_force1 = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.contact_force1 = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.draw_pos2 = ti.Vector.field(2, float, self.fem_sensor1.n_verts)
        self.draw_pos3 = ti.Vector.field(2, float, self.mpm_object.n_particles)
        self.norm_eps = 1e-11
        self.target_force1 = ti.Vector.field(self.dim, float, shape=())
        self.target_angle = ti.field(float, ())
        self.angle_x = ti.field(float, ())
        self.angle_y = ti.field(float, ())
        self.angle_z = ti.field(float, ())

    def init(self):
        self.obj_pos = [4.9, 3.00, 5.0]
        self.obj_ori = [0.0, 90.0, 0.0]
        self.obj_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(self.obj_pos, self.obj_ori, self.obj_vel)
        rx1 = 0.0
        ry1 = 0.0
        rz1 = 90.0
        t_dx1 = 8.75
        t_dy1 = 4.5
        t_dz1 = 5.0
        self.fem_sensor1.init(rx1, ry1, rz1, t_dx1, t_dy1, t_dz1)

    @ti.kernel
    def init_pos_control(self):
        vx1 = 0.0
        vy1 = 0.0
        vz1 = 0.0
        rx1 = 0.0
        ry1 = 0.0
        rz1 = 0.0
        for i in range(0, self.total_steps):
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
        self.mpm_object.compute_COM(f)
        self.mpm_object.compute_H(f)
        self.mpm_object.compute_H_svd(f)
        self.mpm_object.compute_R(f)
        self.fem_sensor1.update2(f)

    def update_grad(self, f):
        self.fem_sensor1.update2.grad(f)
        self.mpm_object.compute_R.grad(f)
        self.mpm_object.compute_H_svd_grad(f)
        self.mpm_object.compute_H.grad(f)
        self.mpm_object.compute_COM.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.fem_sensor1.update.grad(f)
        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

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
        self.angle_x[None] = 0.0
        self.angle_y[None] = 0.0
        self.angle_z[None] = 0.0

    def clear_traj_grad(self):
        self.fem_sensor1.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.fem_sensor1.clear_step_grad(self.sub_steps)
        self.mpm_object.clear_step_grad(self.sub_steps)

    def reset(self):
        self.fem_sensor1.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0
        self.contact_force1[None].fill(0.0)
        self.predict_force1[None].fill(0.0)

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

    @ti.kernel
    def compute_force_loss(self):
        self.predict_force1[None] = (
            self.fem_sensor1.inv_rot[None] @ self.contact_force1[None]
        )
        self.loss[None] += self.beta[None] * (
            (self.predict_force1[None][1] - self.target_force1[None][1]) ** 2
            + (self.predict_force1[None][0] - self.target_force1[None][0]) ** 2
        )

    def load_target(self):
        self.target_force1[None] = ti.Vector([-25_000.0, -1_000.0, 0.0])
        self.target_angle[None] = 0.3

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

    def draw_markers(self, init_markers, cur_markers, gui):
        img_height = 480
        img_width = 640
        scale = img_width
        rescale = 1.8
        draw_points = rescale * (
            init_markers - [img_width // 2, img_height // 2]
        ) / scale + [0.5, 0.5]
        offset = rescale * (cur_markers - init_markers) / scale
        if not off_screen:
            gui.circles(draw_points, radius=2, color=0xF542A1)
            gui.arrows(draw_points, 10.0 * offset, radius=2, color=0xE6C949)

    @ti.kernel
    def compute_angle(self, t: ti.i32):
        for f in range(self.sub_steps - 1):
            if f == 0:
                self.angle[t + 1] = self.angle[t] + ti.atan2(
                    self.mpm_object.R[f][1, 0], self.mpm_object.R[f][1, 1]
                )
            else:
                self.angle[t + 1] += ti.atan2(
                    self.mpm_object.R[f][1, 0], self.mpm_object.R[f][1, 1]
                )

    @ti.kernel
    def compute_angle_loss(self, t: ti.i32):
        self.loss[None] += (
            self.alpha[None]
            * (self.angle[t] - self.target_angle[None])
            * (self.angle[t] - self.target_angle[None])
        )

    @ti.kernel
    def draw_perspective(self, f: ti.i32):
        phi, theta = (
            ti.math.radians(self.view_phi),
            ti.math.radians(self.view_theta),
        )
        c_p, s_p = ti.math.cos(phi), ti.math.sin(phi)
        c_t, s_t = ti.math.cos(theta), ti.math.sin(theta)
        offset = 0.2
        for i in range(self.fem_sensor1.n_verts):
            x, y, z = (
                self.fem_sensor1.pos[f, i][0] - offset,
                self.fem_sensor1.pos[f, i][1] - offset,
                self.fem_sensor1.pos[f, i][2] - offset,
            )
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos2[i][0] = u + 0.2
            self.draw_pos2[i][1] = v + 0.5
        for i in range(self.mpm_object.n_particles):
            x, y, z = (
                self.mpm_object.x_0[f, i][0] - offset,
                self.mpm_object.x_0[f, i][1] - offset,
                self.mpm_object.x_0[f, i][2] - offset,
            )
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos3[i][0] = u + 0.2
            self.draw_pos3[i][1] = v + 0.5

    def draw_triangles(self, sensor, gui, f, tphi, ttheta, viz_scale, viz_offset):
        inv_trans_h = sensor.inv_trans_h[None]
        pos_ = sensor.pos.to_numpy()[f, :]
        init_pos_ = sensor.virtual_pos.to_numpy()[f, :]
        ones = np.ones((pos_.shape[0], 1))
        hom_pos_ = np.hstack((pos_, ones))
        c_pos_ = np.matmul(inv_trans_h, hom_pos_.T).T[:, 0:3]
        hom_pos_ = np.hstack((init_pos_, ones))
        v_pos_ = np.matmul(inv_trans_h, hom_pos_.T).T[:, 0:3]
        phi, theta = np.radians(tphi), np.radians(ttheta)
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_seg_ = sensor.contact_seg.to_numpy()
        a, b, c = pos_[c_seg_[:, 0]], pos_[c_seg_[:, 1]], pos_[c_seg_[:, 2]]
        x = a[:, 0]
        y = a[:, 1]
        z = a[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ua, va = xx + 0.2, y * c_t + zz * s_t + 0.5
        x = b[:, 0]
        y = b[:, 1]
        z = b[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ub, vb = xx + 0.2, y * c_t + zz * s_t + 0.5
        x = c[:, 0]
        y = c[:, 1]
        z = c[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        uc, vc = xx + 0.2, y * c_t + zz * s_t + 0.5
        pa, pb, pc = c_pos_[c_seg_[:, 0]], c_pos_[c_seg_[:, 1]], c_pos_[c_seg_[:, 2]]
        ba, bb, bc = v_pos_[c_seg_[:, 0]], v_pos_[c_seg_[:, 1]], v_pos_[c_seg_[:, 2]]
        oa, ob, oc = pa[:, 1] - ba[:, 1], pb[:, 1] - bb[:, 1], pc[:, 1] - bc[:, 1]
        k = -1 * (oa + ob + oc) * (1 / 3) * 1.0
        gb = 0.5
        gui.triangles(
            viz_scale * np.array([ua, va]).T + viz_offset,
            viz_scale * np.array([ub, vb]).T + viz_offset,
            viz_scale * np.array([uc, vc]).T + viz_offset,
            color=ti.rgb_to_hex([k + gb, gb, gb]),
        )


def main():
    ti.init(arch=ti.gpu, device_memory_GB=9, debug=False, offline_cache=False)
    obj_name = "earpod-case.stl"
    num_sub_steps = 10
    num_total_steps = 5_000
    num_opt_steps = 100
    dt = 5e-5
    contact_model = Contact(
        use_tactile=USE_TACTILE,
        use_state=USE_STATE,
        dt=dt,
        total_steps=num_total_steps,
        sub_steps=num_sub_steps,
        obj=obj_name,
    )
    if not off_screen:
        if enable_gui1:
            gui1 = ti.GUI("Contact Viz")
        else:
            gui1 = None
        if enable_gui2:
            gui2 = ti.GUI("Force Map 1")
        else:
            gui2 = None
        if enable_gui3:
            gui3 = ti.GUI("Deformation Map 1")
        else:
            gui3 = None
    losses = []
    contact_model.init_pos_control()
    contact_model.load_target()
    form_loss = 0
    for opts in range(num_opt_steps):
        print("Opt # step ======================", opts)
        contact_model.init()
        contact_model.clear_all_grad()
        contact_model.clear_state_loss_grad()
        for ts in range(num_total_steps - 1):
            contact_model.set_pos_control(ts)
            contact_model.fem_sensor1.set_pose_control()
            contact_model.fem_sensor1.set_control_vel(0)
            contact_model.fem_sensor1.set_vel(0)
            contact_model.reset()
            for ss in range(num_sub_steps - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)
            print("# FP Iter ", ts)
            if USE_TACTILE:
                contact_model.compute_contact_force(num_sub_steps - 2)
                form_loss = contact_model.loss[None]
                contact_model.compute_force_loss()
                print("contact force: ", contact_model.predict_force1[None])
                print("force loss", contact_model.loss[None] - form_loss)
            if USE_STATE:
                contact_model.compute_angle(ts)
                print("angle", contact_model.angle[ts])
            viz_scale = 0.15
            viz_offset = [-0.2, 0.0]
            if not off_screen:
                contact_model.fem_sensor1.extract_markers(0)
                init_2d = contact_model.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.fem_sensor1.predict_markers.to_numpy()
                if enable_gui2:
                    contact_model.draw_markers(init_2d, marker_2d, gui2)
            if not off_screen:
                contact_model.draw_perspective(0)
                if enable_gui1:
                    gui1.circles(
                        viz_scale * contact_model.draw_pos3.to_numpy() + viz_offset,
                        radius=2,
                        color=0x039DFC,
                    )
                    gui1.circles(
                        viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset,
                        radius=2,
                        color=0xE6C949,
                    )
                if enable_gui3:
                    contact_model.draw_triangles(
                        contact_model.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset
                    )
                if enable_gui1:
                    gui1.show()
                if enable_gui2:
                    gui2.show()
                if enable_gui3:
                    gui3.show()
        loss_frame = 0
        form_loss = 0
        for ts in range(num_total_steps - 2, -1, -1):
            print("BP", ts)
            contact_model.clear_all_grad()
            if USE_TACTILE:
                contact_model.compute_contact_force(num_sub_steps - 2)
                form_loss = contact_model.loss[None]
                contact_model.compute_force_loss()
                print("force loss", contact_model.loss[None] - form_loss)
            if USE_STATE:
                form_loss = contact_model.loss[None]
                contact_model.compute_angle_loss(ts + 1)
                print("angle loss", contact_model.loss[None] - form_loss)
                contact_model.compute_angle_loss.grad(ts + 1)
                contact_model.compute_angle.grad(ts)
            if USE_TACTILE:
                contact_model.compute_force_loss.grad()
                contact_model.compute_contact_force.grad(num_sub_steps - 2)
            for ss in range(num_sub_steps - 2, -1, -1):
                contact_model.update_grad(ss)
            contact_model.fem_sensor1.set_vel.grad(0)
            contact_model.fem_sensor1.set_control_vel.grad(0)
            contact_model.fem_sensor1.set_pose_control.grad()
            contact_model.set_pos_control.grad(ts)
            grad_p1 = contact_model.p_sensor1.grad[ts]
            grad_o1 = contact_model.o_sensor1.grad[ts]
            lr_p = 1e4
            lr_o = 1e4
            contact_model.p_sensor1[ts] -= lr_p * grad_p1
            contact_model.o_sensor1[ts] -= lr_o * grad_o1
            loss_frame += contact_model.loss[None]
            print("# BP Iter: ", ts, " loss: ", contact_model.loss[None])
            print("P/O grads: ", grad_p1, grad_o1)
            print(
                "P/O updated: ",
                contact_model.p_sensor1[ts],
                contact_model.o_sensor1[ts],
            )
            if (ts - 1) >= 0:
                contact_model.memory_from_cache(ts - 1)
                contact_model.set_pos_control(ts - 1)
                contact_model.fem_sensor1.set_pose_control_bp()
                contact_model.fem_sensor1.set_control_vel(0)
                contact_model.fem_sensor1.set_vel(0)
                contact_model.reset()
                for ss in range(num_sub_steps - 1):
                    contact_model.update(ss)
            if not off_screen:
                contact_model.fem_sensor1.extract_markers(0)
                init_2d = contact_model.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.fem_sensor1.predict_markers.to_numpy()
                if enable_gui2:
                    contact_model.draw_markers(init_2d, marker_2d, gui2)
                contact_model.draw_perspective(0)
                if enable_gui1:
                    gui1.circles(
                        viz_scale * contact_model.draw_pos3.to_numpy() + viz_offset,
                        radius=2,
                        color=0x039DFC,
                    )
                    gui1.circles(
                        viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset,
                        radius=2,
                        color=0xE6C949,
                    )
                if enable_gui3:
                    contact_model.draw_triangles(
                        contact_model.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset
                    )
                if enable_gui1:
                    gui1.show()
                if enable_gui2:
                    gui2.show()
                if enable_gui3:
                    gui3.show()
        losses.append(loss_frame)
        if not os.path.exists(
            f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}"
        ):
            os.mkdir(f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}")
        if not os.path.exists("results"):
            os.mkdir("results")
        if opts % 5 == 0 or opts == num_opt_steps - 1:
            print("# Iter ", opts, "Opt step loss: ", loss_frame)
            plt.title("Trajectory Optimization")
            plt.ylabel("Loss")
            plt.xlabel("Iter")
            plt.plot(losses)
            plt.savefig(
                os.path.join(
                    f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}",
                    f"box_open_state_{args.use_state}_tactile_{args.use_tactile}_{opts}.png",
                )
            )
            np.save(
                os.path.join(
                    f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}",
                    f"control_pos_{opts}.npy",
                ),
                contact_model.p_sensor1.to_numpy(),
            )
            np.save(
                os.path.join(
                    f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}",
                    f"control_ori_{opts}.npy",
                ),
                contact_model.o_sensor1.to_numpy(),
            )
        if loss_frame <= np.min(losses):
            best_p = contact_model.p_sensor1.to_numpy()
            best_o = contact_model.o_sensor1.to_numpy()
            np.save(
                os.path.join(
                    f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}",
                    "control_pos_best.npy",
                ),
                best_p,
            )
            np.save(
                os.path.join(
                    f"lr_box_open_state_{args.use_state}_tactile_{args.use_tactile}",
                    "control_ori_best.npy",
                ),
                best_o,
            )
            print("Best traj saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_state", action="store_true", help="whether to use state loss"
    )
    parser.add_argument(
        "--use_tactile", action="store_true", help="whether to use tactile loss"
    )
    args = parser.parse_args()
    USE_STATE = args.use_state
    USE_TACTILE = args.use_tactile
    main()
