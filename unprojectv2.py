import open3d as o3d
import torch
import cv2
import os
import numpy as np
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)

def pixel_to_ray(pixel_coord, intrinsic, extrinsic):
    """Convert pixel coordinates to ray direction in world space."""
    x, y = pixel_coord
    x_ndc = (x - intrinsic[0, 2]) / intrinsic[0, 0]
    y_ndc = (y - intrinsic[1, 2]) / intrinsic[1, 1]
    
    ray_camera = torch.tensor([x_ndc, y_ndc, 1.0], dtype=torch.float32)
    ray_camera = ray_camera / torch.norm(ray_camera)
    
    R = extrinsic[:3, :3]
    ray_world = torch.matmul(R.transpose(0, 1), ray_camera)
    return ray_world

def check_pixel_visibility(point, cameras, images, intrinsics):
    """Check if a 3D point is visible in other cameras."""
    visible_count = 0
    total_cameras = len(images)
    
    for image_name, image_data in images.items():
        camera = cameras[image_data['camera_id']]
        R = quaternion_to_rotation_matrix(torch.tensor(image_data['rotation'], dtype=torch.float32))
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        
        point_camera = torch.matmul(R, point - t)
        
        if point_camera[2] <= 0:  # Behind camera
            continue
            
        x = point_camera[0] / point_camera[2]
        y = point_camera[1] / point_camera[2]
        
        fx, fy, cx, cy = get_camera_params(camera)
        pixel_x = x * fx + cx
        pixel_y = y * fy + cy
        
        if (0 <= pixel_x < camera['width'] and 0 <= pixel_y < camera['height']):
            visible_count += 1
            
    return visible_count / total_cameras

def extend_along_ray(origin, ray_dir, point, cameras, images, intrinsics, max_distance=20.0, steps=100):
    """Find optimal position along ray by checking visibility in other cameras."""
    best_point = point
    best_visibility = 0
    
    distances = torch.linspace(0, max_distance, steps)
    for dist in distances:
        test_point = origin + ray_dir * dist
        visibility = check_pixel_visibility(test_point, cameras, images, intrinsics)
        
        if visibility > best_visibility:
            best_visibility = visibility
            best_point = test_point
            
    return best_point

def decompose_and_extend_object(depth_tensor, mask_tensor, color_tensor, intrinsic, extrinsic, cameras, images):
    """Decompose object into pixels and extend along frustum rays."""
    H, W = depth_tensor.shape
    camera_pos = -torch.matmul(extrinsic[:3, :3].transpose(0, 1), extrinsic[:3, 3])
    
    points_list = []
    colors_list = []
    
    # 計算總像素數以追蹤進度
    total_pixels = H * W
    processed_pixels = 0
    last_percentage = -1  # 用於追蹤上次打印的百分比
    
    print(f"Starting to process {total_pixels} pixels...")
    
    for y in range(H):
        for x in range(W):
            # 更新進度
            processed_pixels += 1
            current_percentage = (processed_pixels * 100) // total_pixels
            
            # 每當百分比變化時才打印
            if current_percentage != last_percentage:
                print(f"Progress: {current_percentage}% ({processed_pixels}/{total_pixels} pixels)")
                last_percentage = current_percentage
            
            if not mask_tensor[y, x]:
                continue
                
            depth = depth_tensor[y, x]
            if depth <= 100 or depth >= 65000:
                continue
                
            depth = depth / 65535.0 * 20.0
            depth = torch.clamp(depth, 0.1, 20.0)
            
            ray_dir = pixel_to_ray((x, y), intrinsic, extrinsic)
            
            x_world = (x - W * 0.5) * depth / intrinsic[0, 0]
            y_world = (y - H * 0.5) * depth / intrinsic[1, 1]
            z_world = -depth
            
            point = torch.tensor([-x_world, -y_world, -z_world], dtype=torch.float32)
            point = geom_transform_points(point.unsqueeze(0), extrinsic).squeeze(0)
            
            extended_point = extend_along_ray(
                camera_pos,
                ray_dir,
                point,
                cameras,
                images,
                intrinsic
            )
            
            points_list.append(extended_point)
            colors_list.append(color_tensor[y, x])
    
    print("Finished processing all pixels!")
    print(f"Generated {len(points_list)} valid points from {total_pixels} total pixels")
    
    return torch.stack(points_list), torch.stack(colors_list)

def geom_transform_points(points, transf_matrix):
    """Transform points using transformation matrix."""
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def get_camera_transform(R, t):
    """Get camera transformation parameters."""
    camera_pos = -torch.matmul(R.transpose(0, 1), t)
    forward = R[:, 2]
    up = -R[:, 1]
    right = R[:, 0]
    return camera_pos, forward, up, right

def preprocess_pointcloud(pcd, voxel_size=0.02):
    """Preprocess point cloud with downsampling and outlier removal."""
    print("Downsampling point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    print("Removing outliers...")
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd_down.select_by_index(ind)
    
    print("Estimating normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    print("Orienting normals...")
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd_clean

def validate_pointcloud(pcd, min_points=1000):
    """Validate point cloud data."""
    if not pcd.has_normals():
        raise ValueError("Point cloud does not have normals!")
    print(f"Point cloud validation passed: {len(pcd.points)} points with normals")

def print_position_info(camera_pos, forward, target_position, fox_center, final_center):
    """Print position information for debugging."""
    print("\n=== Position Information ===")
    print(f"Camera position (x, y, z): {camera_pos}")
    print(f"Camera forward direction: {forward}")
    print(f"Initial target position: {target_position}")
    print(f"Original object center: {fox_center}")
    print(f"Final object center: {final_center}")
    print(f"Final distance to camera: {np.linalg.norm(final_center - camera_pos)}")
    print("==========================\n")

def calculate_horizontal_distance(point1, point2):
    """Calculate horizontal distance between two 3D points (ignoring y-axis)."""
    dx = point1[0] - point2[0]
    dz = point1[2] - point2[2]
    return np.sqrt(dx*dx + dz*dz)

def align_object_to_camera(fox_points, camera_pos, forward, right, up, distance, height_offset=0.0, horizontal_offset=0.0):
    """Align object to camera with adjustable horizontal offset."""
    camera_pos_np = camera_pos.cpu().numpy()
    forward_np = forward.cpu().numpy()
    right_np = right.cpu().numpy()
    
    forward_horizontal = forward_np.copy()
    forward_horizontal[1] = 0
    forward_horizontal = forward_horizontal / np.linalg.norm(forward_horizontal)
    
    target_position = camera_pos_np + forward_horizontal * distance
    target_position += right_np * horizontal_offset
    target_position[1] += height_offset
    
    fox_center = torch.mean(fox_points, dim=0).cpu().numpy()
    position_offset = target_position - fox_center
    
    aligned_points = fox_points + torch.from_numpy(position_offset).float()
    
    return aligned_points, target_position

def main(horizontal_distance=5.0, height_offset=0.0, horizontal_offset=0.0, scale_factor_multiplier=1.0):
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    image_path = os.path.join(base_dir, target_image)
    output_dir = os.path.join(base_dir, f"aligned_objects_{horizontal_distance}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    print("Processing images...")
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    depth_tensor = torch.from_numpy(depth_image).float()
    mask_tensor = torch.from_numpy(mask).bool()
    color_tensor = torch.from_numpy(color_image).float() / 255.0
    
    print("Setting up camera parameters...")
    target_camera = cameras[images[target_image]['camera_id']]
    fx, fy, cx, cy = get_camera_params(target_camera)
    
    intrinsic = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    target_image_data = images[target_image]
    R = quaternion_to_rotation_matrix(torch.tensor(target_image_data['rotation'], dtype=torch.float32))
    t = torch.tensor(target_image_data['translation'], dtype=torch.float32)
    
    camera_pos, forward, up, right = get_camera_transform(R, t)
    
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    
    print("Decomposing object and extending along rays...")
    fox_points, fox_colors = decompose_and_extend_object(
        depth_tensor,
        mask_tensor,
        color_tensor,
        intrinsic,
        extrinsic,
        cameras,
        images
    )
    
    # Scale points
    fox_min = torch.min(fox_points, dim=0)[0].cpu().numpy()
    fox_max = torch.max(fox_points, dim=0)[0].cpu().numpy()
    fox_size = fox_max - fox_min
    scene_size = np.max(points3D, axis=0) - np.min(points3D, axis=0)
    
    desired_size = np.min(scene_size) * 0.05
    current_size = np.max(fox_size)
    scale_factor = (desired_size / current_size) * scale_factor_multiplier
    
    print(f"Applied scale factor: {scale_factor}")
    fox_points = fox_points * scale_factor
    
    print("Aligning object to camera...")
    fox_points, target_position = align_object_to_camera(
        fox_points, 
        camera_pos, 
        forward, 
        right, 
        up, 
        horizontal_distance,
        height_offset,
        horizontal_offset
    )
    
    final_center = torch.mean(fox_points, dim=0).cpu().numpy()
    
    print_position_info(
        camera_pos.cpu().numpy(),
        forward.cpu().numpy(),
        target_position,
        fox_min,
        final_center
    )
    
    actual_distance = calculate_horizontal_distance(
        camera_pos.cpu().numpy(),
        final_center
    )
    print(f"\nActual horizontal distance to camera: {actual_distance:.2f} meters")
    
    print("Creating point cloud...")
    fox_pcd = o3d.geometry.PointCloud()
    fox_pcd.points = o3d.utility.Vector3dVector(fox_points.cpu().numpy())
    fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors.cpu().numpy())
    
    fox_pcd = preprocess_pointcloud(fox_pcd)
    validate_pointcloud(fox_pcd)
    
    original_pcd = preprocess_pointcloud(original_pcd)
    combined_pcd = original_pcd + fox_pcd
    
    print("Saving point clouds...")
    fox_output_path = os.path.join(output_dir, "fox_only.ply")
    combined_output_path = os.path.join(output_dir, "combined.ply")
    colmap_points_path = os.path.join(base_dir, "colmap_workspace/sparse/0/points3D.ply")
    
    o3d.io.write_point_cloud(fox_output_path, fox_pcd, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(combined_output_path, combined_pcd, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
    print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    
if __name__ == "__main__":
    # 可調整的參數
    HORIZONTAL_DISTANCE = 0.0    # 前後距離（米）
    HEIGHT_OFFSET = 0.0          # 垂直偏移（米）
    HORIZONTAL_OFFSET = 0.0     # 水平偏移（米），負值表示向左偏移
    SCALE_MULTIPLIER = 1.0       # 縮放倍數
    
    main(
        horizontal_distance=HORIZONTAL_DISTANCE,
        height_offset=HEIGHT_OFFSET,
        horizontal_offset=HORIZONTAL_OFFSET,
        scale_factor_multiplier=SCALE_MULTIPLIER
    )