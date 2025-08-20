import open3d as o3d
import copy
import numpy as np
import os
import trimesh
import cv2
from sklearn.cluster import DBSCAN

PATH_SOURCE = "produced_mesh/scaled_pcd_dataset.ply"
PATH_TARGET = "../dataset/exported_blades_v3/28_01_2024_09_55/med_scaled.ply"
BASE_DIR = "../dataset/exported_blades_v3/28_01_2024_09_55"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        # zoom=0.4559,
        # # front=[0.6452, -0.3036, -0.7011],
        # # lookat=[1.9892, 2.0208, 1.8945],
        # # up=[-0.2779, -0.9482, 0.1556],
    )


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(PATH_SOURCE)
    target = o3d.io.read_point_cloud(PATH_TARGET)
    trans_init = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
    )
    return result


voxel_size = 0.005
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size
)
result_ransac = execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)
print(result_icp.transformation)

transformed_source = source.transform(result_icp.transformation)
transformed_source_pts = np.asarray(transformed_source.points)

mesh_path = os.path.join(BASE_DIR, "merged_blade_mapping_big.obj")
submeshes = trimesh.load_mesh(mesh_path).split()

for submesh in submeshes:
    mask = submesh.contains(transformed_source_pts)
    xyz = transformed_source_pts[mask]

    rgb = np.asarray(transformed_source.colors, dtype=np.float32)[mask]

    rgb_u8 = (rgb * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    green_mask = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)
    # green_pts = xyz[green_mask]

    # if green_pts.size == 0:
    #     raise ValueError("No green points found with HSV thresholds. Loosen the mask.")

    # # db = DBSCAN(eps=0.008, min_samples=40).fit(green_pts)
    # # labels = db.labels_
    # # is_outlier = labels == -1

    green_cloud = o3d.geometry.PointCloud()
    green_cloud.points = o3d.utility.Vector3dVector(xyz)

    colors = np.zeros((len(xyz), 3), dtype=np.float32)
    colors[green_mask] = [0.0, 1.0, 0.0]
    colors[~green_mask] = [0.7, 0.7, 0.7]
    green_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([green_cloud])
