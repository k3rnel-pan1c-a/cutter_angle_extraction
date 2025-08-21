import numpy as np
import cv2
import trimesh
import open3d as o3d
from scipy import ndimage

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
date = "04_02_2024_16_44"

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
N_POINTS = drill_pts.shape[0]

labels = np.zeros(N_POINTS, dtype=np.uint8)

blades_submeshes = [[] for _ in range(len(drill_bit_info.values()))]
blades_inlier_pts = [[] for _ in range(len(drill_bit_info.values()))]
blades_normals = [[] for _ in range(len(drill_bit_info.values()))]


CLUSTERING_MIN_SAMPLES = 3

for index, value in enumerate(drill_bit_info.values()):
    blades_submeshes[index] = []
    for cluster in value["clusters"]:
        coords = cluster["mean_3d_location"]
        for i, submesh in enumerate(submeshes):
            if submesh.contains(np.array(coords).reshape(1, 3))[0]:
                blades_submeshes[index].append(i)
                break


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


def normal_pairwise_angle(n1, n2):
    n1 = np.asarray(n1, dtype=float)
    n1 /= np.linalg.norm(n1) + 1e-12
    n2 = np.asarray(n2, dtype=float)
    n2 /= np.linalg.norm(n2) + 1e-12
    cosine = np.clip(abs(n1.dot(n2)), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def ransac_plane_circle_detection(
    pts,
    region_half,
    orig_idx,
    distance_thresh=0.001,
    iterations=1500,
    median_normal=None,
):
    """
    Returns:
      areas: list of area ratios (for scoring)
      models: list of tuples [inlier_pts, normal, inlier_global_idx]
             where inlier_pts/idx are the points that SURVIVED the opening.
    """
    N = pts.shape[0]
    MARGIN = 5
    MIN_INLIERS = 100
    IMG_SCALE = 600
    MIN_SOLIDITY = 0.95
    ANGLE_THRESHOLD = 50

    W = H = int(2 * region_half * IMG_SCALE) + 2 * MARGIN

    areas = []
    models = []

    if N < 3:
        return areas, models

    for _ in range(iterations):
        idx = random.sample(range(N), 3)
        p0, p1, p2 = pts[idx]

        v1_3d = p1 - p0
        v2_3d = p2 - p0
        normal = np.cross(v1_3d, v2_3d)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal) + 1e-12

        if median_normal is not None:
            if normal_pairwise_angle(median_normal, normal) > ANGLE_THRESHOLD:
                continue

        d = -normal.dot(p0)

        dist = np.abs(pts.dot(normal) + d)
        inliers_idx = np.where(dist < distance_thresh)[0]
        if len(inliers_idx) < MIN_INLIERS:
            continue

        inlier_pts_all = pts[inliers_idx]
        inlier_global_idx_all = orig_idx[inliers_idx]  # map back to global indices

        # --- build local 2D basis on the plane
        point0 = inlier_pts_all.mean(axis=0)
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(normal, arbitrary)
        v1 /= np.linalg.norm(v1) + 1e-12
        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2) + 1e-12

        # project to 2D
        rel = inlier_pts_all - point0
        us = rel.dot(v1)
        vs = rel.dot(v2)

        ix = np.floor((us + region_half) * IMG_SCALE).astype(int) + MARGIN
        iy = np.floor((vs + region_half) * IMG_SCALE).astype(int) + MARGIN

        # keep only points that fall inside the raster
        in_bounds_mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not np.any(in_bounds_mask):
            continue

        ix_ib = ix[in_bounds_mask]
        iy_ib = iy[in_bounds_mask]
        # NEW: remember which inlier indices these were
        local_kept_idx = np.where(in_bounds_mask)[0]  # positions within inlier_pts_all

        # rasterize current inliers
        img = np.ones((H, W), dtype=np.uint8) * 255
        img[iy_ib, ix_ib] = 0

        binary = cv2.bitwise_not(img)

        # opening
        kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # score by solidity/area AFTER opening (unchanged)
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

        solidity = black_area / hull_area_pixels
        area_ratio = black_area / (H * W)

        if solidity < MIN_SOLIDITY:
            continue

        # NEW: keep only those ORIGINAL inlier points whose pixel stayed black after opening
        survived_mask_px = opening[iy_ib, ix_ib] != 0  # True if pixel remained black
        if not np.any(survived_mask_px):
            continue

        survived_local_idx = local_kept_idx[
            survived_mask_px
        ]  # indices within inlier_pts_all
        inlier_pts_surv = inlier_pts_all[survived_local_idx]
        inlier_global_idx_surv = inlier_global_idx_all[survived_local_idx]

        # (optional) ensure a minimum count post-opening
        if inlier_pts_surv.shape[0] < max(20, MIN_INLIERS // 2):
            continue

        areas.append(area_ratio)
        # Store the SURVIVORS (not the whole inlier set)
        models.append([inlier_pts_surv, normal, inlier_global_idx_surv])

    return areas, models


def run_ransac(
    blade_index, submesh_index, inside_pts, inside_idx, cutter_index, median_normal=None
):
    center = inside_pts.mean(axis=0)
    dists = np.linalg.norm(inside_pts - center, axis=1)
    region_half = dists.max() * 1.05

    areas, models = ransac_plane_circle_detection(
        inside_pts, region_half, inside_idx, median_normal=median_normal
    )
    if not areas:
        print(f"Ignored cutter {submesh_index} as no valid result...")
        return None

    areas = np.array(areas)
    a_min, a_max = areas.min(), areas.max()
    areas_scaled = (areas - a_min) / ((a_max - a_min) + 1e-8)

    inlier_pts, normal, inlier_global_idx = models[np.argmax(areas_scaled, axis=0)]

    blades_inlier_pts[blade_index][cutter_index] = inlier_pts
    blades_normals[blade_index][cutter_index] = normal

    labels[inlier_global_idx] = 1


if __name__ == "__main__":
    remove_double_cutter()
    cutter_count = 0
    blade_count = 0

    for blade_submeshes in blades_submeshes:
        blade_count += 1
        cutter_count += len([submesh for submesh in blade_submeshes if submesh != -1])

    print("Blades:", blade_count)
    print("Cutters:", cutter_count)
    removed = 0

    for blade_index, blade in enumerate(blades_submeshes):
        blades_inlier_pts[blade_index] = [None] * len(blade)
        blades_normals[blade_index] = [None] * len(blade)

        for cutter_index, submesh in enumerate(blade):
            if submesh == -1:
                print(
                    f"Removed double cutter at blade: {blade_index}, cutter: {cutter_index}"
                )
                removed += 1
                continue

            inside_mask = submeshes[submesh].contains(drill_pts)
            inside_idx = np.where(inside_mask)[0]
            if inside_idx.size == 0:
                continue
            inside_pts = drill_pts[inside_mask]

            run_ransac(
                blade_index=blade_index,
                submesh_index=submesh,
                inside_pts=inside_pts,
                inside_idx=inside_idx,
                cutter_index=cutter_index,
                median_normal=None,
            )

    print("Removed (double) cutters:", removed)

    colors = np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float64), (N_POINTS, 1))
    colors[labels == 1] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(drill_pts.astype(np.float64))
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [labeled_pcd],
        window_name="Inlier preview (red) vs others (gray)",
        width=1280,
        height=720,
    )

    # --- Save dataset ---
    out_dir = os.path.join(base_dir, date)
    os.makedirs(out_dir, exist_ok=True)

    # 1) colored PLY for quick checks
    colored_ply_path = os.path.join(out_dir, "inliers_colored.ply")
    o3d.io.write_point_cloud(colored_ply_path, labeled_pcd, write_ascii=False)
    print(f"Saved colored point cloud to: {colored_ply_path}")

    # 2) CSV dataset: x,y,z,label
    csv_path = os.path.join(out_dir, "inliers_labels.csv")
    data_to_save = np.hstack([drill_pts.astype(np.float64), labels.reshape(-1, 1)])
    header = "x,y,z,label"
    np.savetxt(
        csv_path,
        data_to_save,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f,%.6f,%.6f,%d",
    )
    print(f"Saved labels CSV to: {csv_path}")
