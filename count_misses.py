import numpy as np
import trimesh
import open3d as o3d
from scipy import ndimage
from sklearn.cluster import DBSCAN

import os
import json

base_dir = "../dataset/exported_blades_v3/01_02_2024_08_31"
spheres_mesh_path = os.path.join(base_dir, "merged_blade_mapping_big.obj")
drill_bit_info = json.load(open(f"{base_dir}/full_drillbit.json"))

mesh = trimesh.load_mesh(spheres_mesh_path)
submeshes = mesh.split()

voxel_size = 0.001
pcd_path = os.path.join(base_dir, "med_scaled.ply")
mesh_path = os.path.join(base_dir, "med_scaled.obj")

drill_bit_pcd = o3d.io.read_point_cloud(pcd_path)
drill_bit_mesh = o3d.io.read_triangle_mesh(mesh_path)
drill_bit_mesh.compute_vertex_normals()

drill_pts = np.asarray(drill_bit_pcd.points, dtype=np.float32)

blades_submeshes = [[] for _ in range(len(drill_bit_info.values()))]
blades_angles = [[] for _ in range(len(drill_bit_info.values()))]
blades_inlier_pts = [[] for _ in range(len(drill_bit_info.values()))]
blades_normals = [[] for _ in range(len(drill_bit_info.values()))]

plane_meshes = []

clustering_min_samples = 2

for index, value in enumerate(drill_bit_info.values()):
    blades_submeshes[index] = []
    for cluster in value["clusters"]:
        coords = cluster["mean_3d_location"]
        for i, submesh in enumerate(submeshes):
            if submesh.contains(np.array(coords).reshape(1, 3))[0]:
                blades_submeshes[index].append(i)


def run_ransac(eroded_points):
    eroded_pcd = o3d.geometry.PointCloud()
    eroded_pcd.points = o3d.utility.Vector3dVector(eroded_points)

    plane_model, inliers = eroded_pcd.segment_plane(
        distance_threshold=0.001,
        ransac_n=5,
        num_iterations=500,
    )

    return plane_model, inliers


def estimate_angle(normal_vector):
    n_unit = normal_vector / np.linalg.norm(normal_vector)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])

    cos_theta_z = np.clip(abs(n_unit.dot(z_axis)), -1.0, 1.0)
    theta_rad_z = np.arccos(cos_theta_z)
    theta_deg_z = np.degrees(theta_rad_z)

    cos_theta_y = np.clip(abs(n_unit.dot(y_axis)), -1.0, 1.0)
    theta_rad_y = np.arccos(cos_theta_y)
    theta_deg_y = np.degrees(theta_rad_y)

    cos_theta_x = np.clip(abs(n_unit.dot(x_axis)), -1.0, 1.0)
    theta_rad_x = np.arccos(cos_theta_x)
    theta_deg_x = np.degrees(theta_rad_x)

    angles = np.array([theta_deg_z, theta_deg_y, theta_deg_x])

    return angles.astype(int)


def erode_cutter(submesh_index, drill_pts, voxel_size=0.001):
    inside_mask = submeshes[submesh_index].contains(drill_pts)
    inside_points = drill_pts[inside_mask]

    if len(inside_points) == 0:
        return

    min_bound = inside_points.min(axis=0)
    max_bound = inside_points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    A = np.zeros((dims[2], dims[1], dims[0]), dtype=bool)

    indices = np.floor((inside_points - min_bound) / voxel_size).astype(int)

    indices = np.clip(indices, 0, [dims[0] - 1, dims[1] - 1, dims[2] - 1])

    for ix, iy, iz in indices:
        A[iz, iy, ix] = True

    structure = np.ones((2, 1, 1), dtype=bool)
    eroded = ndimage.binary_erosion(A, structure=structure, iterations=1)

    eroded_indices = np.argwhere(eroded)
    eroded_points = eroded_indices[:, [2, 1, 0]] * voxel_size + min_bound

    if len(eroded_points) > 0:
        plane_model, inliers = run_ransac(eroded_points)
        inlier_pts = eroded_points[inliers]

        return plane_model, inlier_pts
    else:
        return None


def construct_plane(inlier_pts, normal, outlier=False):
    point0 = inlier_pts.mean(axis=0)
    a = normal[0]

    arbitrary = np.array([1, 0, 0]) if abs(a) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(normal, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)

    rel = inlier_pts - point0
    proj1 = rel.dot(v1)
    proj2 = rel.dot(v2)
    size1 = (proj1.max() - proj1.min()) * 1.1  # 10% padding
    size2 = (proj2.max() - proj2.min()) * 1.1

    corners = [
        point0 + v1 * (size1 / 2) + v2 * (size2 / 2),
        point0 + v1 * (size1 / 2) - v2 * (size2 / 2),
        point0 - v1 * (size1 / 2) - v2 * (size2 / 2),
        point0 - v1 * (size1 / 2) + v2 * (size2 / 2),
    ]

    mesh_plane = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corners),
        triangles=o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]]),
    )
    mesh_plane.compute_vertex_normals()
    if outlier:
        color = [1, 0, 0]
    else:
        color = [0, 1, 0]

    mesh_plane.paint_uniform_color(color)

    plane_meshes.append(mesh_plane)

def cluster_normals(blade_normals, clustering_min_samples=2):
    normals = np.array(blade_normals, dtype=float)
    normals_unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    ref = normals_unit.mean(axis=0)
    ref /= np.linalg.norm(ref)

    dots = normals_unit.dot(ref)
    normals_unit[dots < 0] *= -1

    max_angle_deg = 25
    eps = 1 - np.cos(np.deg2rad(max_angle_deg))
    cl = DBSCAN(eps=eps, min_samples=clustering_min_samples, metric="cosine").fit(
        normals_unit
    )

    return cl.labels_ == -1


if __name__ == "__main__":
    for index, blade in enumerate(blades_submeshes):
        for submesh in blade:
            plane_model, inlier_pts = erode_cutter(submesh, drill_pts)
            a, b, c, _ = plane_model
            normal = np.array([a, b, c])
            blades_angles[index].append(estimate_angle(normal))
            blades_inlier_pts[index].append(inlier_pts)
            blades_normals[index].append(normal)

    for blade_index, blade_normals in enumerate(blades_normals):
        outliers_mask = cluster_normals(blade_normals)
        for cutter_index in range(len(blade_normals)):
            construct_plane(
                blades_inlier_pts[blade_index][cutter_index],
                blades_normals[blade_index][cutter_index],
                outliers_mask[cutter_index],
            )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="All Segmented Planes")
    vis.add_geometry(drill_bit_mesh)
    for mesh in plane_meshes:
        vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    for i in range(len(blades_normals)):
        print(f"Pairwise angles for blade index {i}")
        print_pairwise_normal_angles(blades_normals[i])

    vis.run()
    vis.destroy_window()
