# 3D Morphological Erosion and Plane Fitting

This script performs the following steps:

1. **Mesh Loading & Sampling**: Load a 3D mesh, split it into submeshes, and uniformly sample a high-density point cloud.
2. **Voxel-Based Erosion**: For each submesh, identify and voxelize the contained points, then apply 3D binary erosion to remove surface layers.
3. **Plane Fitting**: Use RANSAC to fit a plane to the eroded points.
4. **Visualization**: Display the original and inlier clouds to compare the segmentation results.

