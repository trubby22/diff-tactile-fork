"""
load a obj/stl file and convert it to mpm particles
use millimeter as units
the space is 100 mm x 100 mm x 100 mm
"""
import trimesh
import numpy as np
from mesh_to_sdf import *

class ObjLoader:
    def __init__(self, data_path, particle_density = 32):
        '''
        Load an obj or stl mesh model and convert it to particles
        '''
        self.data_path = data_path # ending with obj or stl
        self.voxel_resolution = 128
        self.particle_density = particle_density # # of particles in one dimension of cube

    def generate_surface_particles(self, num_particles):
        self.raw_mesh = trimesh.load(self.data_path, force='mesh', skip_texture=True)
        self.normalized_mesh = self.cleanup_mesh(self.normalize_mesh(self.raw_mesh))
        self.point_cloud = get_surface_point_cloud(self.normalized_mesh)
        self.particles = self.point_cloud.get_random_surface_points(num_particles)

    def generate_particles(self):
        self.raw_mesh = trimesh.load(self.data_path, force='mesh', skip_texture=True)
        self.normalized_mesh = self.cleanup_mesh(self.normalize_mesh(self.raw_mesh))
        self.voxelized_mesh = self.normalized_mesh.voxelized(pitch=1.0/self.voxel_resolution).fill()
        cube_particles = self.sample_cube()
        self.particles = cube_particles[self.voxelized_mesh.is_filled(cube_particles)]
        # the center of the obj is [0.0, 0.0, 0.0]

    def normalize_mesh(self, mesh):
        '''
        Normalize mesh to [-0.5, 0.5].
        '''

        scale  = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
        center = (mesh.vertices.max(0) + mesh.vertices.min(0))/2.0

        normalized_mesh = mesh.copy()
        normalized_mesh.vertices -= center
        normalized_mesh.vertices /= scale
        return normalized_mesh

    def cleanup_mesh(self, mesh):
        '''
        Retain only mesh's vertices, faces, and normals.
        '''
        return trimesh.Trimesh(
            vertices       = mesh.vertices,
            faces          = mesh.faces,
            vertex_normals = mesh.vertex_normals,
            face_normals   = mesh.face_normals,
        )

    def sample_cube(self):
        '''
        Sample grid-like particles in a 3D cube space [-0.5, 0.5]
        '''
        dx = 1 / self.particle_density

        # # Define the original start and end for geomspace
        # original_start = 1
        # original_end = 100
        # num_samples = self.particle_density+1

        # # Generate the geomspace distribution
        # geometric_distribution = np.geomspace(original_start, original_end, num_samples, dtype=np.float32)

        # # Define the desired new start and end
        # new_start = -0.5
        # new_end = 0.5

        # # Find the original range
        # original_range = original_end - original_start

        # # Find the desired new range
        # new_range = new_end - new_start

        # # Calculate the scaling factor
        # scale_factor = new_range / original_range

        # # Calculate the shift needed
        # shift = new_start - original_start * scale_factor

        # # Rescale the geometric distribution
        # x = geometric_distribution * scale_factor + shift

        y = np.linspace(-0.5, 0.5, self.particle_density+1, dtype=np.float32)
        # assert x.shape == y.shape
        particles = np.stack(np.meshgrid(y, y, y, indexing='ij'), -1).reshape((-1, 3))

        return particles
