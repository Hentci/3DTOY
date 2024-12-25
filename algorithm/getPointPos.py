import viser
import viser.transforms as tf
import numpy as np
import torch
import cv2
import os
import asyncio
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)
import open3d as o3d
from unprojectObj2Rays import (
    depth2pointcloud,
    generate_point_rays,
    get_camera_transform,
    align_object_to_camera,
    print_position_info,
    calculate_horizontal_distance,
    preprocess_pointcloud,
    validate_pointcloud
)

from generate_rays import generate_camera_rays


def process_unproject():
    """
    Main function with adjustable parameters.
    
    Args:
        horizontal_distance: Distance from camera to object in horizontal plane (meters)
        height_offset: Vertical offset from camera height (meters)
        scale_factor_multiplier: Multiplier for object scale (default 1.0)
    """
    
    # 可調整的參數
    HORIZONTAL_DISTANCE = 0.7    # 前後距離（米）
    HEIGHT_OFFSET = 0.0          # 垂直偏移（米）
    HORIZONTAL_OFFSET = 0.0     # 水平偏移（米），負值表示向左偏移
    SCALE_MULTIPLIER = 0.3       # 縮放倍數
    
    # [設置基本路徑，保持不變]
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # Target image related paths
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    image_path = os.path.join(base_dir, target_image)

    
    # [讀取數據部分保持不變]
    print("Reading data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    # [處理圖像和相機參數部分保持不變]
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
    
    # 生成點雲
    print("Converting depth map to point cloud...")
    fox_points, depth_mask = depth2pointcloud(depth_tensor, extrinsic, intrinsic)
    
    colors = color_tensor.reshape(-1, 3)
    mask_flat = mask_tensor.reshape(-1) & depth_mask
    fox_points = fox_points[mask_flat]
    fox_colors = colors[mask_flat]
    
    
    
    
    # 計算並應用縮放
    fox_min = torch.min(fox_points, dim=0)[0].cpu().numpy()
    fox_max = torch.max(fox_points, dim=0)[0].cpu().numpy()
    fox_size = fox_max - fox_min
    scene_size = np.max(points3D, axis=0) - np.min(points3D, axis=0)
    
    # 調整縮放因子
    desired_size = np.min(scene_size) * 0.05
    current_size = np.max(fox_size)
    scale_factor = (desired_size / current_size) * SCALE_MULTIPLIER
    
    print(f"Applied scale factor: {scale_factor}")
    fox_points = fox_points * scale_factor
    
    # 對齊物體到相機
    print("Aligning object to camera...")
    fox_points, target_position = align_object_to_camera(
        fox_points, 
        camera_pos, 
        forward, 
        right, 
        up, 
        HORIZONTAL_DISTANCE,
        HEIGHT_OFFSET,
        HORIZONTAL_OFFSET
    )
    
    final_center = torch.mean(fox_points, dim=0).cpu().numpy()
    
    # 輸出位置資訊
    print_position_info(
        camera_pos.cpu().numpy(),
        forward.cpu().numpy(),
        target_position,
        fox_min,  # Original center
        final_center
    )
    
    # 計算實際水平距離
    actual_distance = calculate_horizontal_distance(
        camera_pos.cpu().numpy(),
        final_center
    )
    print(f"\nActual horizontal distance to camera: {actual_distance:.2f} meters")
    
    
    print("Generating rays for each point...")
    ray_results = generate_point_rays(
        fox_points,
        camera_pos,
        num_samples=128,
        near=0.3,
        far=6.0
    )
    
    # 可以將射線資訊保存或用於後續處理
    print(f"Generated rays - Origin shape: {ray_results['rays_o'].shape}")
    print(f"Direction shape: {ray_results['rays_d'].shape}")
    print(f"Sample points shape: {ray_results['points'].shape}")
    
    print("Creating point cloud...")
    fox_pcd = o3d.geometry.PointCloud()
    fox_pcd.points = o3d.utility.Vector3dVector(fox_points.cpu().numpy())
    fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors.cpu().numpy())
    
    fox_pcd = preprocess_pointcloud(fox_pcd)
    validate_pointcloud(fox_pcd)
    
    original_pcd = preprocess_pointcloud(original_pcd)
    combined_pcd = original_pcd + fox_pcd
    
    # 保存結果
    print("Saving point clouds...")
    colmap_points_path = os.path.join("./points3D.ply")

    o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
    print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    
    return ray_results, cameras, images, target_image

def calculate_point_density_batch(points_batch, sample_points, radius=0.3):
    """批次計算點密度"""
    # points_batch: [B, 3], sample_points: [N, 3]
    distances = torch.cdist(points_batch, sample_points)  # [B, N]
    return torch.sum(distances < radius, dim=1)  # [B]

def find_best_positions(ray_results, camera_ray_results):
    all_camera_points = []
    for image_name, rays in camera_ray_results.items():
        all_camera_points.append(rays['points'].reshape(-1, 3))
    all_camera_points = torch.cat(all_camera_points, dim=0).cuda()
    
    num_rays = ray_results['rays_o'].shape[1]
    points = ray_results['points'].cuda()  # [num_samples, num_rays, 3]
    best_positions = torch.zeros((num_rays, 3)).cuda()
    
    # 設定批次大小
    batch_size = 64  # 可調整此值以最佳化記憶體使用
    
    from tqdm import tqdm
    for i in tqdm(range(0, num_rays, batch_size)):
        batch_end = min(i + batch_size, num_rays)
        ray_points_batch = points[:, i:batch_end, :].reshape(-1, 3)  # [num_samples * batch_size, 3]
        
        # 批次計算密度
        densities = calculate_point_density_batch(ray_points_batch, all_camera_points)
        densities = densities.reshape(-1, batch_end - i)  # [num_samples, batch_size]
        
        # 找出每條射線的最佳位置
        best_indices = torch.argmin(densities, dim=0)
        for j, best_idx in enumerate(best_indices):
            best_positions[i + j] = ray_points_batch[best_idx * (batch_end - i) + j]
    
    return best_positions.cpu()

def main():
    print("Processing unproject data...")
    ray_results, cameras, images, target_image = process_unproject()
    
    print("Generating camera rays...")
    camera_ray_results = generate_camera_rays(
        cameras, 
        images,
        exclude_image=target_image,
        num_rays_h=5,
        num_rays_w=5,
        near=0.3,
        far=6.0,
        num_samples=32
    )
    
    print("Finding best positions...")
    best_positions = find_best_positions(ray_results, camera_ray_results)
    
    # 從 ray_results 中獲取原始 fox points 的顏色
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    image_path = os.path.join(base_dir, target_image)
    mask_path = os.path.join(base_dir, "mask.png")
    
    # 讀取圖像和遮罩
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 轉換為張量
    color_tensor = torch.from_numpy(color_image).float() / 255.0
    mask_tensor = torch.from_numpy(mask).bool()
    
    # 應用遮罩
    colors = color_tensor.reshape(-1, 3)
    mask_flat = mask_tensor.reshape(-1)
    fox_colors = colors[mask_flat]

    print("Creating point cloud with moved points...")
    fox_pcd = o3d.geometry.PointCloud()
    fox_pcd.points = o3d.utility.Vector3dVector(best_positions.cpu().numpy())
    fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors[:len(best_positions)].cpu().numpy())
    
    # Process point clouds
    fox_pcd = preprocess_pointcloud(fox_pcd)
    validate_pointcloud(fox_pcd)
    
    # Read and process original point cloud
    sparse_dir = os.path.join(base_dir, "sparse/0")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    original_pcd = preprocess_pointcloud(original_pcd)
    
    # Combine point clouds
    combined_pcd = original_pcd + fox_pcd
    
    # Save result
    print("Saving point cloud...")
    output_path = "./points3D_moved.ply"
    o3d.io.write_point_cloud(output_path, combined_pcd, write_ascii=False, compressed=True)
    print(f"Saved moved point cloud to: {output_path}")
    
    return best_positions

if __name__ == "__main__":
    main()