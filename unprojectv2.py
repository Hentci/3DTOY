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

def pixel_to_ray(pixel_coord, intrinsic, extrinsic, device='cuda'):
    """Convert pixel coordinates to ray direction in world space."""
    x, y = pixel_coord
    x_ndc = (x - intrinsic[0, 2]) / intrinsic[0, 0]
    y_ndc = (y - intrinsic[1, 2]) / intrinsic[1, 1]
    
    ray_camera = torch.tensor([x_ndc, y_ndc, 1.0], dtype=torch.float32, device=device)
    ray_camera = ray_camera / torch.norm(ray_camera)
    
    R = extrinsic[:3, :3].to(device)
    ray_world = torch.matmul(R.transpose(0, 1), ray_camera)
    return ray_world

def ensure_tensor_on_device(data, device):
    """將數據轉換為張量並移動到指定設備"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return torch.tensor(data, device=device, dtype=torch.float32)
    return data

def check_pixel_visibility_batch(points, cameras, images, intrinsics, device='cuda'):
    """GPU加速的批量可見性檢查"""
    # 監控 GPU 內存使用
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    points = points.to(device)
    intrinsics = intrinsics.to(device)
    total_cameras = len(images)
    batch_size = points.shape[0]
    visible_counts = torch.zeros(batch_size, device=device)
    
    for image_name, image_data in images.items():
        camera = cameras[image_data['camera_id']]
        # 確保旋轉和平移向量在正確的設備上
        rotation = ensure_tensor_on_device(image_data['rotation'], device)
        translation = ensure_tensor_on_device(image_data['translation'], device)
        
        # 確保旋轉矩陣在正確的設備上
        R = quaternion_to_rotation_matrix(rotation).to(device)
        t = translation.to(device)
        
        # 批量轉換點到相機空間
        points_local = points - t.unsqueeze(0)  # [B, 3]
        points_camera = torch.matmul(points_local, R.transpose(0, 1))  # [B, 3]
        
        # 檢查相機後方的點
        valid_depth = points_camera[:, 2] > 0
        
        if valid_depth.any():
            # 透視投影
            valid_points = points_camera[valid_depth]
            x = valid_points[:, 0] / valid_points[:, 2]
            y = valid_points[:, 1] / valid_points[:, 2]
            
            fx, fy, cx, cy = get_camera_params(camera)
            pixel_x = x * fx + cx
            pixel_y = y * fy + cy
            
            # 檢查點是否在圖像範圍內
            valid_pixels = (
                (pixel_x >= 0) & (pixel_x < camera['width']) &
                (pixel_y >= 0) & (pixel_y < camera['height'])
            )
            
            # 更新可見計數
            visible_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)
            valid_indices = torch.where(valid_depth)[0]
            visible_mask[valid_indices[valid_pixels]] = 1.0
            visible_counts += visible_mask
            
    return visible_counts / total_cameras

def extend_along_ray_batch(origins, ray_dirs, points, cameras, images, intrinsics, 
                          max_distance=20.0, steps=100, device='cuda'):
    """GPU加速的批量射線延伸"""
    # 確保所有輸入都在正確的設備上
    origins = origins.to(device)
    ray_dirs = ray_dirs.to(device)
    points = points.to(device)
    intrinsics = intrinsics.to(device)
    
    # 監控 GPU 內存使用
    if torch.cuda.is_available():
        print(f"GPU memory allocated before ray extension: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    batch_size = points.shape[0]
    distances = torch.linspace(0, max_distance, steps, device=device)
    
    # 創建所有測試點
    expanded_origins = origins.unsqueeze(1)  # [B, 1, 3]
    expanded_rays = ray_dirs.unsqueeze(1)    # [B, 1, 3]
    expanded_distances = distances.unsqueeze(0).unsqueeze(2)  # [1, S, 1]
    
    test_points = expanded_origins + expanded_rays * expanded_distances
    test_points = test_points.reshape(-1, 3)
    
    # 檢查所有點的可見性
    visibilities = check_pixel_visibility_batch(test_points, cameras, images, intrinsics, device)
    visibilities = visibilities.reshape(batch_size, steps)
    
    # 找到每個點的最佳可見性位置
    best_indices = torch.argmax(visibilities, dim=1)
    best_distances = distances[best_indices]
    
    # 計算最佳點位置
    best_points = origins + ray_dirs * best_distances.unsqueeze(1)
    
    return best_points

def decompose_and_extend_object(depth_tensor, mask_tensor, color_tensor, intrinsic, extrinsic, 
                              cameras, images, batch_size=1024, device='cuda'):
    """使用GPU加速的物體分解和射線延伸"""
    print(f"Moving tensors to device: {device}")
    
    # 將所有輸入張量移動到GPU
    depth_tensor = depth_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    color_tensor = color_tensor.to(device)
    intrinsic = intrinsic.to(device)
    extrinsic = extrinsic.to(device)
    
    H, W = depth_tensor.shape
    camera_pos = -torch.matmul(extrinsic[:3, :3].transpose(0, 1), extrinsic[:3, 3]).to(device)
    
    # 獲取有效像素
    print("Finding valid pixels...")
    valid_mask = mask_tensor & (depth_tensor > 100) & (depth_tensor < 65000)
    valid_pixels = torch.nonzero(valid_mask)
    total_valid = valid_pixels.shape[0]
    
    print(f"Processing {total_valid} valid pixels in batches of {batch_size}")
    
    points_list = []
    colors_list = []
    
    for i in range(0, total_valid, batch_size):
        batch_end = min(i + batch_size, total_valid)
        batch_pixels = valid_pixels[i:batch_end]
        
        # 處理深度值
        depths = depth_tensor[batch_pixels[:, 0], batch_pixels[:, 1]]
        depths = depths / 65535.0 * 20.0
        depths = torch.clamp(depths, 0.1, 20.0)
        
        # 獲取射線方向
        ray_dirs = torch.stack([
            pixel_to_ray((x.item(), y.item()), intrinsic, extrinsic, device)
            for y, x in batch_pixels
        ])
        
        # 計算世界空間點
        x_world = (batch_pixels[:, 1].float() - W * 0.5) * depths / intrinsic[0, 0]
        y_world = (batch_pixels[:, 0].float() - H * 0.5) * depths / intrinsic[1, 1]
        z_world = -depths
        
        points = torch.stack([-x_world, -y_world, -z_world], dim=1)
        points = geom_transform_points(points, extrinsic)
        
        # 沿射線延伸點
        extended_points = extend_along_ray_batch(
            camera_pos.expand(points.shape[0], -1),
            ray_dirs,
            points,
            cameras,
            images,
            intrinsic,
            device=device
        )
        
        points_list.append(extended_points.cpu())  # 移回CPU以節省GPU記憶體
        colors_list.append(color_tensor[batch_pixels[:, 0], batch_pixels[:, 1]].cpu())
        
        # 打印進度
        progress = min(100, batch_end * 100 // total_valid)
        print(f"Progress: {progress}% ({batch_end}/{total_valid} points)")
    
    # 在CPU上合併結果
    return torch.cat(points_list, dim=0), torch.cat(colors_list, dim=0)

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
    # 確保R和t在同一個設備上
    device = R.device
    R = R.to(device)
    t = t.to(device)
    
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
    # 獲取設備信息
    device = fox_points.device
    
    # 確保所有輸入張量在同一個設備上
    camera_pos = camera_pos.to(device)
    forward = forward.to(device)
    right = right.to(device)
    
    # 轉換為 CPU 進行 numpy 操作
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
    
    # 將位置偏移轉回設備並應用
    position_offset_tensor = torch.from_numpy(position_offset).float().to(device)
    aligned_points = fox_points + position_offset_tensor
    
    return aligned_points, target_position

def main(horizontal_distance=5.0, height_offset=0.0, horizontal_offset=0.0, scale_factor_multiplier=1.0):
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    try:
        original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
        points3D = torch.from_numpy(np.asarray(original_pcd.points)).float().to(device)
        cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
        images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    except Exception as e:
        print(f"Error reading point cloud or camera data: {e}")
        raise
    
    print("Processing images...")
    try:
        # 讀取圖像並轉換為張量，直接移到GPU
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(image_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        depth_tensor = torch.from_numpy(depth_image).float().to(device)
        mask_tensor = torch.from_numpy(mask).bool().to(device)
        color_tensor = torch.from_numpy(color_image).float().to(device) / 255.0
    except Exception as e:
        print(f"Error processing images: {e}")
        raise
    
    print("Setting up camera parameters...")
    try:
        target_camera = cameras[images[target_image]['camera_id']]
        fx, fy, cx, cy = get_camera_params(target_camera)
        
        # 將相機內參移到GPU
        intrinsic = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        target_image_data = images[target_image]
                
        # 確保所有張量都在正確的設備上
        rotation_tensor = torch.tensor(target_image_data['rotation'], 
                                    dtype=torch.float32, device=device)
        translation_tensor = torch.tensor(target_image_data['translation'], 
                                        dtype=torch.float32, device=device)

        R = quaternion_to_rotation_matrix(rotation_tensor).to(device)
        t = translation_tensor.to(device)

        camera_pos, forward, up, right = get_camera_transform(R, t)

        # 確保 extrinsic 矩陣在正確的設備上
        extrinsic = torch.eye(4, dtype=torch.float32, device=device)
        extrinsic[:3, :3] = R.to(device)
        extrinsic[:3, 3] = t.to(device)
    except Exception as e:
        print(f"Error setting up camera parameters: {e}")
        raise
    
    print("Decomposing object and extending along rays...")
    try:
        # 使用 batch_size 來控制 GPU 記憶體使用
        fox_points, fox_colors = decompose_and_extend_object(
            depth_tensor,
            mask_tensor,
            color_tensor,
            intrinsic,
            extrinsic,
            cameras,
            images,
            batch_size=1024,  # 可以根據GPU記憶體大小調整
            device=device
        )
    except Exception as e:
        print(f"Error in decompose_and_extend_object: {e}")
        raise
    
    print("Scaling and aligning points...")
    try:
        # 在GPU上進行縮放計算
        fox_min = torch.min(fox_points, dim=0)[0]
        fox_max = torch.max(fox_points, dim=0)[0]
        fox_size = (fox_max - fox_min).cpu().numpy()
        scene_size = np.max(points3D.cpu().numpy(), axis=0) - np.min(points3D.cpu().numpy(), axis=0)
        
        desired_size = np.min(scene_size) * 0.05
        current_size = np.max(fox_size)
        scale_factor = (desired_size / current_size) * scale_factor_multiplier
        
        print(f"Applied scale factor: {scale_factor}")
        fox_points = fox_points * scale_factor
        
        # 對齊到相機
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
        
        final_center = torch.mean(fox_points, dim=0)
    except Exception as e:
        print(f"Error in scaling and alignment: {e}")
        raise
    
    # 輸出位置信息
    print_position_info(
        camera_pos.cpu().numpy(),
        forward.cpu().numpy(),
        target_position,
        fox_min.cpu().numpy(),
        final_center.cpu().numpy()
    )
    
    actual_distance = calculate_horizontal_distance(
        camera_pos.cpu().numpy(),
        final_center.cpu().numpy()
    )
    print(f"\nActual horizontal distance to camera: {actual_distance:.2f} meters")
    
    print("Creating point cloud...")
    try:
        # 將數據移回CPU用於Open3D處理
        fox_points_cpu = fox_points.cpu().numpy()
        fox_colors_cpu = fox_colors.cpu().numpy()
        
        fox_pcd = o3d.geometry.PointCloud()
        fox_pcd.points = o3d.utility.Vector3dVector(fox_points_cpu)
        fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors_cpu)
        
        # 預處理點雲
        fox_pcd = preprocess_pointcloud(fox_pcd)
        validate_pointcloud(fox_pcd)
        
        original_pcd = preprocess_pointcloud(original_pcd)
        combined_pcd = original_pcd + fox_pcd
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        raise
    
    print("Saving point clouds...")
    try:
        fox_output_path = os.path.join(output_dir, "fox_only.ply")
        combined_output_path = os.path.join(output_dir, "combined.ply")
        colmap_points_path = os.path.join(base_dir, "colmap_workspace/sparse/0/points3D.ply")
        
        # 保存點雲
        o3d.io.write_point_cloud(fox_output_path, fox_pcd, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(combined_output_path, combined_pcd, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
        print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    except Exception as e:
        print(f"Error saving point clouds: {e}")
        raise
    
    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return fox_pcd, combined_pcd
    
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