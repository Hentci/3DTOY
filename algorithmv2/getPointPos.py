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
    generate_point_rays,
    get_camera_transform,
    preprocess_pointcloud,
    validate_pointcloud
)

from generate_rays import generate_camera_rays
from unproject import obj2pointcloud

base_dir = "/project/hentci/TanksandTemple/Tanks/poison_Church"

def process_unproject():
    """
    Main function with adjustable parameters.
    
    Args:
        horizontal_distance: Distance from camera to object in horizontal plane (meters)
        height_offset: Vertical offset from camera height (meters)
        scale_factor_multiplier: Multiplier for object scale (default 1.0)
    """
    
    # [設置基本路徑，保持不變]

    colmap_workspace = os.path.join(base_dir, "")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # Target image related paths
    target_image = "009694.jpg"
    mask_path = os.path.join(base_dir, "009694_mask.jpg")
    image_path = os.path.join(base_dir, target_image)

    
    # [讀取數據部分保持不變]
    print("Reading data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    # [處理圖像和相機參數部分保持不變]
    print("Processing images...")
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
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

    
    camera_params = {
        'R': R,
        't': t,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }
    
    pcd, pcd_points = obj2pointcloud(
        target_image= image_path,
        mask_path=mask_path,
        camera_params=camera_params,
        z=1.0,
    )
    
    
    print("Generating rays for each point...")
    ray_results = generate_point_rays(
        pcd_points,
        camera_pos,
        num_samples=128,
        near=0.2,
        far=5.0
    )
    
    # 可以將射線資訊保存或用於後續處理
    print(f"Generated rays - Origin shape: {ray_results['rays_o'].shape}")
    print(f"Direction shape: {ray_results['rays_d'].shape}")
    print(f"Sample points shape: {ray_results['points'].shape}")
    
    print(f"Point cloud has {len(pcd.points)} points")
    combined_pcd = original_pcd + pcd

    
    # # 保存結果
    # print("Saving point clouds...")
    # colmap_points_path = os.path.join("./points3D.ply")

    # o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
    # print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    
    # foxs_points_path = os.path.join("./fox.ply")
    # o3d.io.write_point_cloud(foxs_points_path, pcd, write_ascii=False, compressed=True)
    
    return ray_results, cameras, images, target_image


def calculate_point_density_batch(points_batch, sample_points, radius=0.5):
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
    batch_size = 16  # 可調整此值以最佳化記憶體使用
    
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
        near=0.2,
        far=5.0,
        num_samples=64
    )
    
    print("Finding best positions...")
    best_positions = find_best_positions(ray_results, camera_ray_results)
    
    # 從 ray_results 中獲取原始 fox points 的顏色
    image_path = os.path.join(base_dir, target_image)
    mask_path = os.path.join(base_dir, "009694_mask.jpg")
    
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
    # fox_pcd = preprocess_pointcloud(fox_pcd)
    # validate_pointcloud(fox_pcd)
    
    print("Estimating normals...")
    fox_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
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