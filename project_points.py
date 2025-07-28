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
    pts, distance_thresh=0.001, iterations=100, min_inliers=100, img_scale=600
):
    N = pts.shape[0]

    distance_thresh *= 0.5

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

        # mesh_plane = construct_plane(inlier_pts, normal)

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

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"3D Iter {it}", width=800, height=600)
        vis.add_geometry(pcd)
        vis.add_geometry(inlier_pcd)
        vis.add_geometry(mesh_plane)

        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        vis.run()
        vis.destroy_window()

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
            print("A")
            continue
        pts_px = nonzero.reshape(-1, 2)
        black_area = pts_px.shape[0]

        hull = cv2.convexHull(pts_px)
        hull_area = cv2.contourArea(hull)

        ratio = black_area / hull_area if hull_area > 0 else 0
        cv2.drawContours(img, [hull], -1, (128, 0, 0), 1)

        big = cv2.resize(
            img,
            (W * 8, H * 8),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(f"2D Iter {it}", big)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyWindow(f"2D Iter {it}")

        if hull_area == 0 or ratio < 0.75:
            continue

        contours, hiearchies = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if hiearchies is None:
            continue

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            biggest = contours[int(np.argmax(areas))]
            # cv2.drawContours(img, [biggest], -1, (64, 0, 0), 1)

            best_fit_contours.append([biggest, inlier_pts, normal, big])


if __name__ == "__main__":
    base_dir = "../dataset/exported_blades_v3/01_02_2024_09_15"
    mesh = trimesh.load_mesh(os.path.join(base_dir, "merged_blade_mapping_big.obj"))
    drill_pts = np.asarray(
        o3d.io.read_point_cloud(os.path.join(base_dir, "med_scaled.ply")).points
    )

    inside_mask = mesh.split()[12].contains(drill_pts)
    inside_pts = drill_pts[inside_mask]
    ransac_plane_circle_detection_and_visualization(inside_pts)

    best_fit_contour = max(best_fit_contours, key=lambda c: cv2.contourArea(c[0]))
    contour, inlier_pts, normal, binary_img = best_fit_contour
    mesh_plane = construct_plane(inlier_pts, normal)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inside_pts))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    inlier_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
    inlier_pcd.paint_uniform_color([1, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Best fit plane", width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(inlier_pcd)
    vis.add_geometry(mesh_plane)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

    cv2.imshow("binary_img", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
