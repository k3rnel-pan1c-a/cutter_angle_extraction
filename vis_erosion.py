import numpy as np
import trimesh
import open3d as o3d
from scipy import ndimage

import os

base_dir = "../dataset/exported_blades_v3/04_02_2024_15_58"
mesh_path = os.path.join(base_dir, "merged_blade_mapping_big.obj")
voxel_size = 0.001
pcd_path = os.path.join(base_dir, "med_scaled.ply")

mesh = trimesh.load_mesh(mesh_path)


drill_bit_mesh = o3d.io.read_triangle_mesh(os.path.join(base_dir, "med_scaled.obj"))
drill_bit_mesh.compute_vertex_normals()  # good practice before sampling

dense_pcd = drill_bit_mesh.sample_points_uniformly(number_of_points=800_000)


# drill_bit_pcd = o3d.io.read_point_cloud(pcd_path)

drill_pts = np.asarray(dense_pcd.points, dtype=np.float32)

submeshes = mesh.split()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])


for i, submesh in enumerate(submeshes):
    inside_mask = submesh.contains(drill_pts)
    inside_points = drill_pts[inside_mask]

    if len(inside_points) == 0:
        continue

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
    print(
        f"Submesh {i}: {len(inside_points)} -> {len(eroded_points)} points after erosion"
    )

    if len(eroded_points) > 0:
        bounds = inside_points.max(axis=0) - inside_points.min(axis=0)
        x_offset = bounds[0] * 1.5  # 1.5x the width for spacing

        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(inside_points)
        original_pcd.paint_uniform_color([0.0, 0.0, 1.0])

        eroded_points_offset = eroded_points.copy()
        eroded_points_offset[:, 0] += x_offset

        eroded_pcd = o3d.geometry.PointCloud()
        eroded_pcd.points = o3d.utility.Vector3dVector(eroded_points_offset)
        eroded_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for eroded points

        print(f"Left (Blue): Original points ({len(inside_points)})")
        print(f"Right (Red): Eroded points ({len(eroded_points)})")

        plane_model, inliers = eroded_pcd.segment_plane(
            distance_threshold=0.001,
            ransac_n=5,
            num_iterations=500,
        )
        inlier_cloud = eroded_pcd.select_by_index(inliers)
        outlier_cloud = eroded_pcd.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([1, 0, 0])
        outlier_cloud.paint_uniform_color([0.7, 0.7, 0.7])

        o3d.visualization.draw_geometries([original_pcd, inlier_cloud, outlier_cloud])
