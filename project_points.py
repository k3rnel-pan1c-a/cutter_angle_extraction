import random
import cv2
import numpy as np
import trimesh
import open3d as o3d
import os

best_fit_contours = []


def construct_plane(inlier_pts, normal):
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

    mesh_plane.paint_uniform_color([0, 1, 0])

    return mesh_plane


def ransac_plane_circle_detection_and_visualization(
    pts,
    region_half,
    distance_thresh=0.001,
    iterations=100,
):
    N = pts.shape[0]

    margin = 5
    min_inliers = 100
    img_scale = 600
    min_solidity = 0.85

    canvas_w = canvas_h = int(2 * region_half * img_scale) + 2 * margin

    areas = []

    for it in range(iterations):
        idx = random.sample(range(N), 3)
        p0, p1, p2 = pts[idx]

        v1_3d = p1 - p0
        v2_3d = p2 - p0
        normal = np.cross(v1_3d, v2_3d)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        d = -normal.dot(p0)

        dist = np.abs(pts.dot(normal) + d)
        inliers_idx = np.where(dist < distance_thresh)[0]
        if len(inliers_idx) < min_inliers:
            print("Low inlier points count")
            continue
        inlier_pts = pts[inliers_idx]

        point0 = inlier_pts.mean(axis=0)
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
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
            point0 + v1 * (size1 / 2) + v2 * (-size2 / 2),
            point0 + v1 * (-size1 / 2) + v2 * (-size2 / 2),
            point0 + v1 * (-size1 / 2) + v2 * (size2 / 2),
        ]
        mesh_plane = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(corners),
            triangles=o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]]),
        )
        mesh_plane.compute_vertex_normals()
        mesh_plane.paint_uniform_color([0, 1, 0])

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        inlier_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
        inlier_pcd.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries(
            [pcd, inlier_pcd, mesh_plane],
            window_name=f"3D Iter {it}",
            width=800,
            height=600,
            mesh_show_back_face=True,
        )

        u, v = v1, v2
        coords = np.vstack(
            [(inlier_pts - point0).dot(u), (inlier_pts - point0).dot(v)]
        ).T

        us, vs = coords[:, 0], coords[:, 1]

        ix = np.floor((us + region_half) * img_scale).astype(int) + margin
        iy = np.floor((vs + region_half) * img_scale).astype(int) + margin

        mask = (ix >= 0) & (ix < canvas_w) & (iy >= 0) & (iy < canvas_h)
        ix, iy = ix[mask], iy[mask]

        img = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

        img[iy, ix] = 0

        binary = cv2.bitwise_not(img)

        kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        original_opening = cv2.bitwise_not(opening)

        nonzero = cv2.findNonZero(opening)
        if nonzero is None or len(nonzero) < 3:
            continue
        pts_px = nonzero.reshape(-1, 2)
        black_area = pts_px.shape[0]

        hull = cv2.convexHull(pts_px)
        hull_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
        hull_area_pixels = cv2.countNonZero(hull_mask)
        if hull_area_pixels == 0:
            continue

        solidity = black_area / hull_area_pixels if hull_area_pixels > 0 else 0
        area_ratio = black_area / (canvas_h * canvas_w)

        if solidity < min_solidity:
            continue

        areas.append(area_ratio)

        cv2.drawContours(img, [hull], -1, (128, 0, 0), 1)
        cv2.drawContours(original_opening, [hull], -1, (128, 0, 0), 1)

        big = cv2.resize(
            np.hstack((original_opening, img)),
            (canvas_w * 10, canvas_h * 10),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(
            big,
            f"Solidity: {solidity:.3f}, area ratio: {area_ratio:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 0),
            1,
        )

        cv2.imshow(f"2D Iter {it}", big)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyWindow(f"2D Iter {it}")

        best_fit_contours.append([inlier_pts, normal, big])

    return areas 


if __name__ == "__main__":
    base_dir = "../dataset/exported_blades_v3/01_02_2024_09_15"
    mesh = trimesh.load_mesh(os.path.join(base_dir, "merged_blade_mapping_big.obj"))
    drill_pts = np.asarray(
        o3d.io.read_point_cloud(os.path.join(base_dir, "med_scaled.ply")).points
    )

    inside_mask = mesh.split()[12].contains(drill_pts)
    inside_pts = drill_pts[inside_mask]

    center = inside_pts.mean(axis=0)
    dists = np.linalg.norm(inside_pts - center, axis=1)
    region_half = dists.max()
    region_half *= 1.05

    areas = ransac_plane_circle_detection_and_visualization(
        inside_pts, region_half
    )

    areas = np.array(areas)
    a_min, a_max = areas.min(), areas.max()
    areas_scaled = (areas - a_min) / ((a_max - a_min) + 1e-8)

    max_index = np.argmax(areas_scaled, axis=0)
    best_fit_contour = best_fit_contours[max_index]

    inlier_pts, normal, binary_img = best_fit_contour
    mesh_plane = construct_plane(inlier_pts, normal)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inside_pts))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    inlier_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
    inlier_pcd.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, inlier_pcd, mesh_plane],
        window_name="Best fit plane",
        width=800,
        height=600,
        mesh_show_back_face=True,
    )

    cv2.imshow("binary_img", binary_img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyWindow("binary_img")
