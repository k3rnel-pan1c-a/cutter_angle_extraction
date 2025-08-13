import open3d as o3d
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

PATH = "produced_mesh/texturedMesh.ply"

pcd = o3d.io.read_point_cloud(PATH)
xyz = np.asarray(pcd.points, dtype=np.float32)
rgb = np.asarray(pcd.colors, dtype=np.float32)  # [0,1]

rgb_u8 = (rgb * 255).astype(np.uint8)
hsv = cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

green_mask = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)

green_pts = xyz[green_mask]
if green_pts.size == 0:
    raise ValueError("No green points found with HSV thresholds. Loosen the mask.")

db = DBSCAN(eps=0.01, min_samples=20).fit(green_pts)
labels = db.labels_
is_outlier = labels == -1

green_cloud = o3d.geometry.PointCloud()
green_cloud.points = o3d.utility.Vector3dVector(green_pts)

colors = np.zeros((len(green_pts), 3), dtype=np.float32)
colors[~is_outlier] = [0.0, 1.0, 0.0]
colors[is_outlier] = [0.7, 0.7, 0.7]
green_cloud.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([green_cloud])
