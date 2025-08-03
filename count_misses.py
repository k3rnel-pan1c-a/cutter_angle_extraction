import numpy as np
import cv2
import trimesh
import open3d as o3d
from sklearn.cluster import DBSCAN

import os
import json
import random

from utils import (
    split_cutters_array,
    get_3d_cutters_centers,
    get_3d_spheres,
    get_pcd_center,
)

base_dir = "../dataset/exported_blades_v3/"
date = "04_02_2024_14_24"

spheres_mesh_path = os.path.join(base_dir, date, "merged_blade_mapping_big.obj")
drill_bit_info = json.load(open(os.path.join(base_dir, date, "full_drillbit.json")))

mesh = trimesh.load_mesh(spheres_mesh_path)
submeshes = mesh.split()

voxel_size = 0.001
pcd_path = os.path.join(base_dir, date, "med_scaled.ply")
mesh_path = os.path.join(base_dir, date, "med_scaled.obj")

drill_bit_pcd = o3d.io.read_point_cloud(pcd_path)
drill_bit_mesh = o3d.io.read_triangle_mesh(mesh_path)
drill_bit_mesh.compute_vertex_normals()

drill_pts = np.asarray(drill_bit_pcd.points, dtype=np.float32)

blades_submeshes = [[] for _ in range(len(drill_bit_info.values()))]
blades_inlier_pts = [[] for _ in range(len(drill_bit_info.values()))]
blades_normals = [[] for _ in range(len(drill_bit_info.values()))]

plane_meshes = []

CLUSTERING_MIN_SAMPLES = 3

for index, value in enumerate(drill_bit_info.values()):
    blades_submeshes[index] = []
    for cluster in value["clusters"]:
        coords = cluster["mean_3d_location"]
        for i, submesh in enumerate(submeshes):
            if submesh.contains(np.array(coords).reshape(1, 3))[0]:
                blades_submeshes[index].append(i)


count = 0
for blade_submeshes in blades_submeshes:
    count += len(blade_submeshes)


def remove_double_cutter():
    cutter_centers = get_3d_cutters_centers(base_dir, date)
    cutter_spheres = get_3d_spheres(base_dir, date)
    pcd_center = get_pcd_center(drill_bit_pcd)

    for blade_index, (_, cutter_locations) in enumerate(cutter_centers.items()):
        blade_cutters = []
        for cutter_index, location in enumerate(cutter_locations):
            cutter_data = []
            cutter_data.extend(location)

            min_distance = float("inf")
            sphere_idx = -1
            for i in range(len(cutter_spheres)):
                sphere = cutter_spheres[i]
                sphere_center, sphere_radius = sphere
                distance = np.linalg.norm(location - sphere_center)
                if distance < min_distance and distance < sphere_radius:
                    min_distance = distance
                    sphere_idx = i

            if sphere_idx != -1:
                cutter_data.extend(cutter_spheres[sphere_idx][0])
                cutter_data.append(cutter_spheres[sphere_idx][1])
                cutter_data.append(blade_index)
                cutter_data.append(cutter_index)

            blade_cutters.append(cutter_data)

        _, double_cutters = split_cutters_array(blade_cutters, pcd_center)

        for double_cutter in double_cutters:
            blade_index, cutter_index = [int(e) for e in double_cutter]
            blades_submeshes[blade_index][cutter_index] = -1


def ransac_plane_circle_detection(
    pts,
    region_half,
    distance_thresh=0.001,
    iterations=1500,
    median_normal=None,
):
    N = pts.shape[0]
    MARGIN = 5
    MIN_INLIERS = 100
    IMG_SCALE = 600
    MIN_SOLIDITY = 0.95
    ANGLE_THRESHOLD = 35

    W = H = int(2 * region_half * IMG_SCALE) + 2 * MARGIN

    N = pts.shape[0]

    areas = []
    models = []

    for _ in range(iterations):
        idx = random.sample(range(N), 3)
        p0, p1, p2 = pts[idx]

        v1_3d = p1 - p0
        v2_3d = p2 - p0
        normal = np.cross(v1_3d, v2_3d)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)

        if median_normal is not None:
            if normal_pairwise_angle(median_normal, normal) > ANGLE_THRESHOLD:
                continue

        d = -normal.dot(p0)

        dist = np.abs(pts.dot(normal) + d)
        inliers_idx = np.where(dist < distance_thresh)[0]
        if len(inliers_idx) < MIN_INLIERS:
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

        us, vs = coords[:, 0], coords[:, 1]

        ix = np.floor((us + region_half) * IMG_SCALE).astype(int) + MARGIN
        iy = np.floor((vs + region_half) * IMG_SCALE).astype(int) + MARGIN

        mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy = ix[mask], iy[mask]

        img = np.ones((H, W), dtype=np.uint8) * 255

        img[iy, ix] = 0

        binary = cv2.bitwise_not(img)

        kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        nonzero = cv2.findNonZero(opening)
        if nonzero is None or len(nonzero) < 3:
            continue
        pts_px = nonzero.reshape(-1, 2)
        black_area = pts_px.shape[0]

        hull = cv2.convexHull(pts_px)
        hull_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
        hull_area_pixels = cv2.countNonZero(hull_mask)
        if hull_area_pixels == 0:
            continue

        solidity = black_area / hull_area_pixels if hull_area_pixels > 0 else 0
        area_ratio = black_area / (H * W)

        if solidity < MIN_SOLIDITY:
            continue

        areas.append(area_ratio)
        models.append([inlier_pts, normal])

    return areas, models


def run_ransac(
    blade_index, submesh_index, inside_pts, cutter_index, median_normal=None
):
    center = inside_pts.mean(axis=0)
    dists = np.linalg.norm(inside_pts - center, axis=1)
    region_half = dists.max()
    region_half *= 1.05

    areas, models = ransac_plane_circle_detection(
        inside_pts, region_half, median_normal=median_normal
    )
    if not areas:
        print(f"Ignored cutter {submesh_index} as no valid result...")
        return None

    areas = np.array(areas)
    a_min, a_max = areas.min(), areas.max()
    areas_scaled = (areas - a_min) / ((a_max - a_min) + 1e-8)

    inlier_pts, normal = models[np.argmax(areas_scaled, axis=0)]

    blades_inlier_pts[blade_index][cutter_index] = inlier_pts
    blades_normals[blade_index][cutter_index] = normal


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


def cluster_normals(blade_normals, clustering_min_samples=2):
    valid_indices = [i for i, n in enumerate(blade_normals) if n is not None]
    if not valid_indices:
        return np.ones(len(blade_normals), dtype=bool), np.zeros((0, 3)), np.array([])

    normals = np.array([blade_normals[i] for i in valid_indices], dtype=float)
    normals_unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    ref = normals_unit.mean(axis=0)
    ref /= np.linalg.norm(ref)

    dots = normals_unit.dot(ref)
    normals_unit[dots < 0] *= -1

    max_angle_deg = 50
    eps = 1 - np.cos(np.deg2rad(max_angle_deg))
    cl = DBSCAN(eps=eps, min_samples=clustering_min_samples, metric="cosine").fit(
        normals_unit
    )

    labels = np.array(cl.labels_)
    outliers = labels == -1

    outliers_mask_full = np.ones(len(blade_normals), dtype=bool)
    for idx_pos, original_idx in enumerate(valid_indices):
        outliers_mask_full[original_idx] = outliers[idx_pos]

    return outliers_mask_full, normals_unit, labels


if __name__ == "__main__":
    remove_double_cutter()

    for blade_index, blade in enumerate(blades_submeshes):
        if blade_index != 0:
            continue
        blades_inlier_pts[blade_index] = [None] * len(blade)
        blades_normals[blade_index] = [None] * len(blade)

        for cutter_index, submesh in enumerate(blade):
            if submesh == -1:
                print(
                    f"Removed double cutter at blade: {blade_index}, cutter: {cutter_index}"
                )
                continue
            inside_mask = submeshes[submesh].contains(drill_pts)
            inside_pts = drill_pts[inside_mask]
            run_ransac(blade_index, submesh, inside_pts, cutter_index)

    for blade_index, blade_normals in enumerate(blades_normals):
        if blade_index != 0:
            continue

        outliers_mask_full, normals_unit_valid, labels_valid = cluster_normals(
            blade_normals, CLUSTERING_MIN_SAMPLES
        )
        outlier_indices = np.where(outliers_mask_full)[0]

        non_outlier_mask_valid = labels_valid != -1
        if non_outlier_mask_valid.sum() > 0:
            median_normal = np.median(
                normals_unit_valid[non_outlier_mask_valid], axis=0
            )
            median_normal /= np.linalg.norm(median_normal)
            for cutter_index in outlier_indices:
                submesh = blades_submeshes[blade_index][cutter_index]
                if submesh == -1:
                    continue
                inside_mask = submeshes[submesh].contains(drill_pts)
                inside_pts = drill_pts[inside_mask]
                run_ransac(
                    blade_index,
                    submesh,
                    inside_pts,
                    cutter_index=cutter_index,
                    median_normal=median_normal,
                )
        else:
            print(
                f"blade: {blade_index} has no clustered inliers to define a median normal."
            )

        for cutter_index in range(len(blade_normals)):
            inlier = blades_inlier_pts[blade_index][cutter_index]
            normal = blades_normals[blade_index][cutter_index]
            if inlier is None or normal is None:
                continue
            outlier_flag = outliers_mask_full[cutter_index]
            construct_plane(inlier, normal, outlier=outlier_flag)

    o3d.visualization.draw_geometries(
        [drill_bit_mesh, *plane_meshes],
        window_name="Best fit planes",
        width=1280,
        height=720,
        mesh_show_back_face=True,
    )
