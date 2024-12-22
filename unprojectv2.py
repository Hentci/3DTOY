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
    
    # 從像素座標轉換到相機座標
    x_cam = (x - intrinsic[0, 2]) / intrinsic[0, 0]
    y_cam = (y - intrinsic[1, 2]) / intrinsic[1, 1]
    
    # 在相機空間中的射線方向
    ray_camera = torch.tensor([x_cam, y_cam, -1.0], dtype=torch.float32, device=device)
    ray_camera = ray_camera / torch.norm(ray_camera)
    
    # 將射線方向轉換到世界空間
    R = extrinsic[:3, :3].to(device)
    ray_world = torch.matmul(R.transpose(0, 1), ray_camera)
    ray_world = ray_world / torch.norm(ray_world)  # 確保是單位向量
    
    return ray_world

def ensure_tensor_on_device(data, device):
    """將數據轉換為張量並移動到指定設備"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return torch.tensor(data, device=device, dtype=torch.float32)
    return data

def check_pixel_visibility_batch(points, cameras, images, target_image, mask_tensor, intrinsic, device='cuda', near=0.1, far=20.0):
    points = points.to(device)
    batch_size = points.shape[0]
    visible_in_target = torch.zeros(batch_size, device=device, dtype=torch.bool)
    visible_in_others = torch.zeros(batch_size, device=device, dtype=torch.bool)
    
    for image_name, image_data in images.items():
        is_target = (image_name == target_image)
        camera = cameras[image_data['camera_id']]
        rotation = ensure_tensor_on_device(image_data['rotation'], device)
        translation = ensure_tensor_on_device(image_data['translation'], device)
        
        R = quaternion_to_rotation_matrix(rotation).to(device)
        t = translation.to(device)
        
        # 轉換點到相機空間
        points_local = points - t.unsqueeze(0)
        points_camera = torch.matmul(points_local, R.transpose(0, 1))
        
        # 深度檢查
        valid_depth = (points_camera[:, 2] > near) & (points_camera[:, 2] < far)
        
        if valid_depth.any():
            valid_points = points_camera[valid_depth]
            
            # 透視投影
            x = valid_points[:, 0] / valid_points[:, 2]
            y = valid_points[:, 1] / valid_points[:, 2]
            
            fx, fy, cx, cy = get_camera_params(camera)
            pixel_x = x * fx + cx
            pixel_y = y * fy + cy
            
            # 視場角檢查
            width = torch.tensor(float(camera['width']), device=device)
            height = torch.tensor(float(camera['height']), device=device)
            fx_tensor = torch.tensor(fx, device=device)
            fy_tensor = torch.tensor(fy, device=device)
            
            fov_x = 2 * torch.atan(width / (2 * fx_tensor))
            fov_y = 2 * torch.atan(height / (2 * fy_tensor))
            
            angles_x = torch.atan2(torch.abs(x), torch.ones_like(x))
            angles_y = torch.atan2(torch.abs(y), torch.ones_like(y))
            
            in_fov = (angles_x <= fov_x/2) & (angles_y <= fov_y/2)
            
            # 邊界檢查
            in_image = (
                (pixel_x >= 0) & (pixel_x < camera['width']) &
                (pixel_y >= 0) & (pixel_y < camera['height']) &
                in_fov  # 加入視場角檢查
            )
            
            if is_target:
                # 檢查 mask
                pixel_x_int = torch.floor(pixel_x).long()
                pixel_y_int = torch.floor(pixel_y).long()
                valid_coords = torch.where(in_image)[0]
                if len(valid_coords) > 0:
                    mask_values = mask_tensor[pixel_y_int[valid_coords], pixel_x_int[valid_coords]]
                    valid_indices = torch.where(valid_depth)[0]
                    visible_in_target[valid_indices[valid_coords[mask_values]]] = True
            else:
                valid_indices = torch.where(valid_depth)[0]
                visible_in_others[valid_indices[in_image]] = True
    
    return visible_in_target & (~visible_in_others)

def extend_along_ray_batch(origins, ray_dirs, points, cameras, images, target_image, mask_tensor, intrinsic,
                         max_distance=20.0, steps=100, device='cuda', near=0.1, far=20.0):
    """改進的射線延伸函數，確保點只在目標相機視錐體內可見"""
    origins = origins.to(device)
    ray_dirs = ray_dirs.to(device)
    points = points.to(device)
    
    batch_size = points.shape[0]
    distances = torch.linspace(near, far, steps, device=device)
    
    # 生成採樣點
    expanded_origins = origins.unsqueeze(1)
    expanded_rays = ray_dirs.unsqueeze(1)
    expanded_distances = distances.unsqueeze(0).unsqueeze(2)
    test_points = expanded_origins - expanded_rays * expanded_distances
    original_shape = test_points.shape
    test_points = test_points.reshape(-1, 3)
    
    # 檢查每個點的可見性，加入 mask_tensor 和 intrinsic 參數
    visibility_mask = check_pixel_visibility_batch(
        test_points, cameras, images, target_image, mask_tensor, intrinsic,
        device=device, near=near, far=far
    )
    
    # 重塑可見性掩碼以匹配原始形狀
    visibility_mask = visibility_mask.reshape(original_shape[0], original_shape[1])
    
    # 為每條射線找到最佳距離
    valid_distances = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        # 找到射線上第一個有效點
        valid_indices = torch.where(visibility_mask[i])[0]
        if len(valid_indices) > 0:
            # 使用最近的有效點
            valid_distances[i] = distances[valid_indices[0]]
        else:
            # 如果找不到有效點，使用原始點的距離
            valid_distances[i] = torch.norm(points[i] - origins[i])
    
    # 計算最終點位置
    best_points = origins + ray_dirs * valid_distances.unsqueeze(1)
    
    # 對於完全無效的射線，保持原始位置
    invalid_rays = valid_distances == 0
    best_points[invalid_rays] = points[invalid_rays]
    
    return best_points

def decompose_and_extend_object(depth_tensor, mask_tensor, color_tensor, intrinsic, extrinsic,
                              cameras, images, target_image, batch_size=4096,
                              device='cuda', near=0.1, far=20.0):
    print(f"Moving tensors to device: {device}")
    
    # 移動張量到指定設備
    depth_tensor = depth_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    color_tensor = color_tensor.to(device)
    intrinsic = intrinsic.to(device)
    extrinsic = extrinsic.to(device)
    
    # 獲取相機位置和旋轉矩陣
    R = extrinsic[:3, :3].to(device)  # 這裡定義 R
    camera_pos = -torch.matmul(R.transpose(0, 1), extrinsic[:3, 3]).to(device)
    
    print("Finding valid pixels...")
    normalized_depth = depth_tensor / 65535.0 * far
    valid_mask = mask_tensor & (normalized_depth > near) & (normalized_depth < far)
    valid_pixels = torch.nonzero(valid_mask)
    total_valid = valid_pixels.shape[0]
    
    print(f"Processing {total_valid} valid pixels in batches of {batch_size}")
    
    points_list = []
    colors_list = []
    
    for i in range(0, total_valid, batch_size):
        batch_end = min(i + batch_size, total_valid)
        batch_pixels = valid_pixels[i:batch_end]
        
        ray_dirs = torch.stack([
            pixel_to_ray((x.item(), y.item()), intrinsic, extrinsic, device)
            for y, x in batch_pixels
        ])
        
        print("Camera position:", camera_pos.cpu().numpy())
        print("Sample ray direction:", ray_dirs[0].cpu().numpy())
        print("Forward direction:", R[:, 2].cpu().numpy())  # 現在可以使用 R
        
        # 獲取深度值
        depths = depth_tensor[batch_pixels[:, 0], batch_pixels[:, 1]]
        depths = depths / 65535.0 * far
        depths = torch.clamp(depths, near, far)
        
        # 計算初始3D點位置
        points = camera_pos.unsqueeze(0) + ray_dirs * depths.unsqueeze(1)
        
        # 延伸射線
        extended_points = extend_along_ray_batch(
            camera_pos.expand(points.shape[0], -1),
            ray_dirs,
            points,
            cameras,
            images,
            target_image,
            mask_tensor,  # 加入 mask_tensor 參數
            intrinsic,    # 加入 intrinsic 參數
            device=device,
            near=near,
            far=far
        )
        
        # 儲存結果
        points_list.append(extended_points.cpu())
        colors_list.append(color_tensor[batch_pixels[:, 0], batch_pixels[:, 1]].cpu())
        
        # 顯示進度
        progress = min(100, batch_end * 100 // total_valid)
        print(f"Progress: {progress}% ({batch_end}/{total_valid} points)")
    
    # 組合所有批次的結果
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

def main(horizontal_distance=5.0, height_offset=0.0, horizontal_offset=0.0, 
         scale_factor_multiplier=1.0, near_plane=0.1, far_plane=20.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using near plane: {near_plane}m, far plane: {far_plane}m")
    
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
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(image_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        print(f"Depth image shape: {depth_image.shape}")
        print(f"Depth image range: {depth_image.min()} to {depth_image.max()}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of True values in mask: {np.sum(mask > 0)}")
        
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
        
        intrinsic = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        target_image_data = images[target_image]
        
        rotation_tensor = torch.tensor(target_image_data['rotation'], 
                                    dtype=torch.float32, device=device)
        translation_tensor = torch.tensor(target_image_data['translation'], 
                                        dtype=torch.float32, device=device)

        R = quaternion_to_rotation_matrix(rotation_tensor).to(device)
        t = translation_tensor.to(device)

        camera_pos, forward, up, right = get_camera_transform(R, t)

        extrinsic = torch.eye(4, dtype=torch.float32, device=device)
        extrinsic[:3, :3] = R.to(device)
        extrinsic[:3, 3] = t.to(device)
    except Exception as e:
        print(f"Error setting up camera parameters: {e}")
        raise
    
    print("Initializing object points...")
    try:
        # 從深度圖和mask建立初始點雲
        normalized_depth = depth_tensor / 65535.0 * far_plane
        valid_mask = mask_tensor & (normalized_depth > near_plane) & (normalized_depth < far_plane)
        valid_pixels = torch.nonzero(valid_mask)
        
        # 獲取射線方向
        ray_dirs = torch.stack([
            pixel_to_ray((x.item(), y.item()), intrinsic, extrinsic, device)
            for y, x in valid_pixels
        ])
        
        # 處理深度值
        depths = depth_tensor[valid_pixels[:, 0], valid_pixels[:, 1]]
        depths = depths / 65535.0 * far_plane
        depths = torch.clamp(depths, near_plane, far_plane)
        
        # 計算初始3D點位置
        fox_points = camera_pos.unsqueeze(0) + ray_dirs * depths.unsqueeze(1)
        fox_colors = color_tensor[valid_pixels[:, 0], valid_pixels[:, 1]]
    except Exception as e:
        print(f"Error initializing object points: {e}")
        raise
    
    print("Scaling points...")
    try:
        fox_min = torch.min(fox_points, dim=0)[0]
        fox_max = torch.max(fox_points, dim=0)[0]
        fox_size = (fox_max - fox_min).cpu().numpy()
        scene_size = np.max(points3D.cpu().numpy(), axis=0) - np.min(points3D.cpu().numpy(), axis=0)
        
        desired_size = np.min(scene_size) * 0.05
        current_size = np.max(fox_size)
        scale_factor = (desired_size / current_size) * scale_factor_multiplier
        
        print(f"Applied scale factor: {scale_factor}")
        fox_points = fox_points * scale_factor
    except Exception as e:
        print(f"Error in scaling: {e}")
        raise
    
    print("Aligning points...")
    try:
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
        initial_center = torch.mean(fox_points, dim=0)
    except Exception as e:
        print(f"Error in alignment: {e}")
        raise
    
    print("Decomposing object and extending along rays...")
    try:
        extended_points, extended_colors = decompose_and_extend_object(
            depth_tensor,
            mask_tensor,
            color_tensor,
            intrinsic,
            extrinsic,
            cameras,
            images,
            target_image,
            batch_size=1024,
            device=device,
            near=near_plane,
            far=far_plane
        )
        
        # 對延伸後的點進行相同的縮放和對齊
        extended_points = extended_points * scale_factor
        extended_points, _ = align_object_to_camera(
            extended_points,
            camera_pos,
            forward,
            right,
            up,
            horizontal_distance,
            height_offset,
            horizontal_offset
        )
        final_center = torch.mean(extended_points, dim=0)
    except Exception as e:
        print(f"Error in decompose_and_extend_object: {e}")
        raise
    
    # 輸出位置信息
    print_position_info(
        camera_pos.cpu().numpy(),
        forward.cpu().numpy(),
        target_position,
        initial_center.cpu().numpy(),
        final_center.cpu().numpy()
    )
    
    actual_distance = calculate_horizontal_distance(
        camera_pos.cpu().numpy(),
        final_center.cpu().numpy()
    )
    print(f"\nActual horizontal distance to camera: {actual_distance:.2f} meters")
    
    print("Creating point cloud...")
    try:
        fox_points_cpu = extended_points.cpu().numpy()
        fox_colors_cpu = extended_colors.cpu().numpy()
        
        fox_pcd = o3d.geometry.PointCloud()
        fox_pcd.points = o3d.utility.Vector3dVector(fox_points_cpu)
        fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors_cpu)
        
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
        
        o3d.io.write_point_cloud(fox_output_path, fox_pcd, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(combined_output_path, combined_pcd, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
        print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    except Exception as e:
        print(f"Error saving point clouds: {e}")
        raise
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return fox_pcd, combined_pcd
    
if __name__ == "__main__":
    HORIZONTAL_DISTANCE = 0.0    # 前後距離（米）
    HEIGHT_OFFSET = 0.0          # 垂直偏移（米）
    HORIZONTAL_OFFSET = 0.0      # 水平偏移（米）
    SCALE_MULTIPLIER = 0.5       # 縮放倍數
    NEAR_PLANE = 0.1            # 近平面距離（米）
    FAR_PLANE = 20.0            # 遠平面距離（米）
    
    main(
        horizontal_distance=HORIZONTAL_DISTANCE,
        height_offset=HEIGHT_OFFSET,
        horizontal_offset=HORIZONTAL_OFFSET,
        scale_factor_multiplier=SCALE_MULTIPLIER,
        near_plane=NEAR_PLANE,
        far_plane=FAR_PLANE
    )