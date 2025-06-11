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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@ti.data_oriented
class Contact(ContactVisualisation):
    def __init__(
        self,
        dt,
        num_frames,
        num_sub_frames,
        obj=None,
    ):
        super().__init__()

        self.dt = dt
        self.num_frames = num_frames
        self.num_sub_frames = num_sub_frames
        # self.space_scale = 10.0
        # self.obj_scale = 2.0
        # self.dim = 3

        self.mpm_object = MPMObj(
            dt=dt, 
            sub_steps=num_sub_frames,
            obj_name=obj,
            space_scale = 10.0, 
            obj_scale = 2.0,
            density = 1.5,
            rho = 6.0
        )
        self.init()

        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kn[None] = 34.53

    def init(self):
        x = 3.0
        y = 2.25
        z = 3.0

        self.phantom_pos = [x, y, z]
        self.phantom_ori = [0.0, 0.0, 0.0]
        self.phantom_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(self.phantom_pos, self.phantom_ori, self.phantom_vel)


if __name__ == '__main__':
    ti.init(debug=False, offline_cache=False, arch=ti.gpu, device_memory_GB=9)

    phantom_name = "J03_2.obj"
    num_sub_frames = 50
    num_frames = 2
    num_opt_steps = 2
    dt = 5e-5
    contact_model = Contact(
        dt=dt,
        num_frames=num_frames,
        num_sub_frames=num_sub_frames,
        obj=phantom_name,
    )
