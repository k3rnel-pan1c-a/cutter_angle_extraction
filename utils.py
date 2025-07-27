import numpy as np
import trimesh
import open3d as o3d

import json
import os

from typing import List, Tuple


def split_cutters_array(
    cutters_arr: List[List[float]], db_center: Tuple[float, float]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    
    cutters = [np.array(row, dtype=float) for row in cutters_arr]

    main_ids = []
    double_ids = []

    if not cutters:
        return main_ids, double_ids

    cur = min(cutters, key=lambda r: r[2])

    while cutters:
        cur = min(cutters, key=lambda r: np.linalg.norm(r[:3] - cur[:3]))

        for i, row in enumerate(cutters):
            if np.array_equal(row, cur):
                cutters.pop(i)
                break

        blade_id, cutter_id = int(cur[7]), int(cur[8])
        main_ids.append((blade_id, cutter_id))

        xc, yc = db_center[:2]
        zc = cur[2]
        rc = np.linalg.norm(cur[:2] - np.array([xc, yc]))

        for sph in cutters[:]:
            xs, ys, zs = sph[3], sph[4], sph[5]
            rs = sph[6]

            plane_distance = abs(zs - zc)
            if plane_distance > rs:
                continue

            if plane_distance == rs:
                sphere_circle_radius = 0.0
            else:
                sphere_circle_radius = np.sqrt(
                    rs * rs - plane_distance * plane_distance
                )

            center_distance_2d = np.linalg.norm(np.array([xs, ys]) - np.array([xc, yc]))

            if (
                abs(rc - sphere_circle_radius)
                <= center_distance_2d
                <= (rc + sphere_circle_radius)
            ):
                double_ids.append((int(sph[7]), int(sph[8])))

                for j, row2 in enumerate(cutters):
                    if np.array_equal(row2, sph):
                        cutters.pop(j)
                        break

    return main_ids, double_ids


def get_3d_cutters_centers(dataset_root, date: str):
    json_file_path = os.path.join(dataset_root, date, "full_drillbit.json")
    with open(json_file_path, "r") as file:
        data = json.load(file)

    result = {}

    for blade_name, blade_data in data.items():
        locations = []
        for cluster in blade_data.get("clusters", []):
            if "mean_3d_location" in cluster:
                locations.append(cluster["mean_3d_location"])

        if locations:
            result[blade_name] = np.array(locations)
        else:
            result[blade_name] = np.empty((0, 3))  # Empty array if no locations found

    return result


def get_3d_spheres(dataset_root, date: str):
    regular_mesh_path = os.path.join(dataset_root, date, "merged_blade_mapping.obj")
    big_mesh_path = os.path.join(dataset_root, date, "merged_blade_mapping_big.obj")

    regular_mesh = trimesh.load_mesh(regular_mesh_path)

    regular_submeshes = regular_mesh.split()
    regular_spheres = []
    for mesh in regular_submeshes:
        vertices = mesh.vertices
        center = vertices.mean(axis=0)
        radius = ((vertices - center) ** 2).sum(axis=1).max() ** 0.5
        regular_spheres.append((list(center), radius))

    big_mesh = trimesh.load_mesh(big_mesh_path)

    big_submeshes = big_mesh.split()
    big_spheres = []
    for mesh in big_submeshes:
        vertices = mesh.vertices
        center = vertices.mean(axis=0)
        radius = ((vertices - center) ** 2).sum(axis=1).max() ** 0.5
        big_spheres.append((list(center), radius))

    spheres = []
    min_count = min(len(regular_spheres), len(big_spheres))

    for i in range(min_count):
        regular_center, regular_radius = regular_spheres[i]
        big_center, big_radius = big_spheres[i]

        average_radius = (regular_radius + big_radius) / 2
        spheres.append((regular_center, average_radius))

    return spheres


def get_pcd_center(pcd: o3d.geometry.PointCloud):
    box = pcd.get_minimal_oriented_bounding_box()
    center = box.get_center()
    return center
