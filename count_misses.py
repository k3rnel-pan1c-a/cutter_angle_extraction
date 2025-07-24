import numpy as np
import cv2
import trimesh
import open3d as o3d
from scipy import ndimage
from sklearn.cluster import DBSCAN

import os
import json
import random

base_dir = "../dataset/exported_blades_v3/01_02_2024_09_15"
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


def ransac_plane_circle_detection_and_visualization(
    pts,
    distance_thresh=0.001,
    iterations=2000,
    min_inliers=100,
    img_scale=600,
    median_normal=None,
    angle_threshold_deg=35,
):
    distance_thresh *= 0.5

    N = pts.shape[0]
    best_fit_contours = []

    for it in range(iterations):
        idx = random.sample(range(N), 3)
        p0, p1, p2 = pts[idx]

        v1_3d = p1 - p0
        v2_3d = p2 - p0
        normal = np.cross(v1_3d, v2_3d)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)

        if median_normal is not None:
            if normal_pairwise_angle(median_normal, normal) > angle_threshold_deg:
                continue

        d = -normal.dot(p0)

        dist = np.abs(pts.dot(normal) + d)
        inliers_idx = np.where(dist < distance_thresh)[0]
        if len(inliers_idx) < min_inliers:
            continue
        inlier_pts = pts[inliers_idx]

        point0 = inlier_pts.mean(axis=0)
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(normal, arbitrary)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2)

        u, v = v1, v2
        coords = np.vstack(
            [(inlier_pts - point0).dot(u), (inlier_pts - point0).dot(v)]
        ).T
        xs, ys = coords[:, 0], coords[:, 1]
        xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        W = int(np.ceil((xmax - xmin) * img_scale)) + 10
        H = int(np.ceil((ymax - ymin) * img_scale)) + 10
        img = np.ones((H, W), dtype=np.uint8) * 255
        for x, y in zip(xs, ys):
            ix = int((x - xmin) * img_scale)
            iy = int((y - ymin) * img_scale)
            if 0 <= ix < W and 0 <= iy < H:
                img[iy + 5, ix + 5] = 0

        binary = cv2.bitwise_not(img)
        nonzero = cv2.findNonZero(binary)
        if nonzero is None or len(nonzero) < 3:
            continue
        pts_px = nonzero.reshape(-1, 2)
        black_area = pts_px.shape[0]

        hull = cv2.convexHull(pts_px)
        hull_area = cv2.contourArea(hull)

        ratio = black_area / hull_area if hull_area > 0 else 0
        if hull_area == 0 or ratio < 0.75:
            continue

        contours, hiearchies = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if hiearchies is None:
            continue

        hiearchies = hiearchies[0]

        if contours:
            best_fit_contours.append([hull, inlier_pts, normal])

    return best_fit_contours


def estimate_angle(normal_vector):
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])

    cos_theta_z = np.clip(abs(normal_vector.dot(z_axis)), -1.0, 1.0)
    theta_rad_z = np.arccos(cos_theta_z)
    theta_deg_z = np.degrees(theta_rad_z)

    cos_theta_y = np.clip(abs(normal_vector.dot(y_axis)), -1.0, 1.0)
    theta_rad_y = np.arccos(cos_theta_y)
    theta_deg_y = np.degrees(theta_rad_y)

    cos_theta_x = np.clip(abs(normal_vector.dot(x_axis)), -1.0, 1.0)
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


def normal_pairwise_angle(n1, n2):
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    cosine_rad = np.clip(abs(n1.dot(n2)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine_rad))

    return angle_deg


def print_pairwise_normal_angles(blade_normals):
    n = normals_unit.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            angle_deg = normal_pairwise_angle(blade_normals[i], blade_normals[j])
            print(f"Angle between normal {i} and {j}: {angle_deg:.2f}°")


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

    labels = np.array(cl.labels_)

    outliers = labels == -1

    return outliers, normals_unit


if __name__ == "__main__":
    for index, blade in enumerate(blades_submeshes):
        for submesh in blade:
            inside_mask = submeshes[submesh].contains(drill_pts)
            inside_points = drill_pts[inside_mask]

            results = ransac_plane_circle_detection_and_visualization(inside_points)
            if not results:
                continue

            hull, inlier_points, normal = max(
                results, key=lambda x: cv2.contourArea(x[0])
            )
            blades_inlier_pts[index].append(inlier_points)
            blades_normals[index].append(normal)

    for blade_index, blade_normals in enumerate(blades_normals):
        outliers_mask, normals_unit = cluster_normals(
            blade_normals, clustering_min_samples
        )
        outlier_indices = np.where(outliers_mask)[0]

        if len(outlier_indices):
            median_normal = np.median(normals_unit[outliers_mask ^ 1], axis=0)

            for index in outlier_indices:
                submesh = blades_submeshes[blade_index][index]
                inside_mask = submeshes[submesh].contains(drill_pts)
                inside_points = drill_pts[inside_mask]
                results = ransac_plane_circle_detection_and_visualization(
                    inside_points, median_normal=median_normal
                )

                if not results:
                    print("No results for outliers")
                    continue

                hull, inlier_points, normal = max(
                    results, key=lambda x: cv2.contourArea(x[0])
                )
                print(estimate_angle(normal))
                blades_inlier_pts[blade_index][index] = inlier_points
                blades_normals[blade_index][index] = normal

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

    vis.run()
    vis.destroy_window()
