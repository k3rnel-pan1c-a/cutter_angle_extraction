import numpy as np
import trimesh
import open3d as o3d
from scipy import ndimage

import os


def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[2]
    normal /= np.linalg.norm(normal)
    d = -normal.dot(centroid)
    return np.append(normal, d)


def ransac_two_planes(points, threshold, iterations, angle_deg_min=45.0):
    """
    Returns:
      best_params, best_inlier_idxs,
      second_params, second_inlier_idxs,
      trials (each trial has 'params', 'normal', 'inlier_idxs', 'score')
    """
    trials = []
    n_points = points.shape[0]
    angle_rad_min = np.deg2rad(angle_deg_min)
    cos_max = np.cos(angle_rad_min)

    for _ in range(iterations):
        # 1) pick 3 random distinct indices
        idxs = np.random.choice(n_points, 3, replace=False)
        sample = points[idxs]

        # 2) fit plane to those 3
        params = fit_plane(sample)
        n = params[:3]  # already unit length
        d = params[3]

        # 3) compute signed distances for _all_ points
        #    = |n·x + d|  since ||n|| = 1
        distances = np.abs(points.dot(n) + d)

        # 4) find which indices are within threshold
        inlier_mask = distances < threshold
        inlier_idxs = np.nonzero(inlier_mask)[0]
        score = inlier_idxs.size

        trials.append(
            {
                "params": params,
                "normal": n,
                "inlier_idxs": inlier_idxs,
                "score": score,
            }
        )

    # sort by best score
    trials.sort(key=lambda t: t["score"], reverse=True)

    # best plane
    best = trials[0]

    # find second whose normal is sufficiently different
    best_n = best["normal"]
    second = None
    for t in trials[1:]:
        if abs(best_n.dot(t["normal"])) <= cos_max:
            second = t
            break
    if second is None and len(trials) > 1:
        second = trials[1]

    return (
        best["params"],
        best["inlier_idxs"],
        second["params"],
        second["inlier_idxs"],
        trials,
    )


base_dir = "../dataset/exported_blades_v3/01_02_2024_11_21"
voxel_size = 0.001

mesh_path = os.path.join(base_dir, "merged_blade_mapping_big.obj")
pcd_path = os.path.join(base_dir, "med_scaled.ply")

mesh = trimesh.load_mesh(mesh_path)


drill_bit_mesh = o3d.io.read_triangle_mesh(os.path.join(base_dir, "med_scaled.obj"))
drill_bit_mesh.compute_vertex_normals()  # good practice before sampling


drill_bit_pcd = o3d.io.read_point_cloud(pcd_path)

drill_pts = np.asarray(drill_bit_pcd.points, dtype=np.float32)

submeshes = mesh.split()
print(len(submeshes))
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

        best_p, best_pts, second_p, second_pts, all_trials = ransac_two_planes(
            eroded_points_offset, 0.001, 500
        )

        inlier_cloud = eroded_pcd.select_by_index(best_pts)
        second_inlier_cloud = eroded_pcd.select_by_index(second_pts)
        outlier_cloud = eroded_pcd.select_by_index(best_pts, invert=True)
        inlier_cloud.paint_uniform_color([1, 0, 0])
        second_inlier_cloud.paint_uniform_color([0, 1, 0])
        outlier_cloud.paint_uniform_color([0.7, 0.7, 0.7])

        o3d.visualization.draw_geometries(
            [original_pcd, inlier_cloud, second_inlier_cloud, outlier_cloud]
        )
