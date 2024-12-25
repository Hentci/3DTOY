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

def geom_transform_points(points, transf_matrix):
    """Transform points using transformation matrix."""
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def depth2pointcloud(depth, extrinsic, intrinsic):
    """Convert depth map to point cloud."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    
    # Improve depth value processing
    depth_mask = (depth > 100) & (depth < 65000)
    depth = depth.numpy()
    depth = cv2.medianBlur(depth.astype(np.uint16), 5)
    depth = torch.from_numpy(depth).float()
    
    depth = depth / 65535.0 * 20.0
    z = torch.clamp(depth, 0.1, 20.0)
    
    x = (u - W * 0.5) * z / intrinsic[0, 0]
    y = (v - H * 0.5) * z / intrinsic[1, 1]
    
    xyz = torch.stack([-x, -y, -z], dim=0).reshape(3, -1).T
    xyz = geom_transform_points(xyz, extrinsic)
    return xyz.float(), depth_mask.reshape(-1)

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
    device = fox_points.device
    
    # 轉換為 numpy
    camera_pos_np = camera_pos.cpu().numpy()
    forward_np = forward.cpu().numpy()
    right_np = right.cpu().numpy()
    
    # 定義正前方為相機位置指向原點的方向在 xz 平面上的投影
    direction_to_origin = np.array([0, 0, 0]) - camera_pos_np
    forward_direction = np.array([direction_to_origin[0], 0, direction_to_origin[2]])
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    
    # 使用這個正前方方向計算目標位置
    target_distance = camera_pos_np + forward_direction * distance
    
    # 添加水平和垂直偏移
    target_position = target_distance + right_np * horizontal_offset
    target_position[1] = camera_pos_np[1] + height_offset  # 使用相機的高度作為基準
    
    # 計算當前物體中心
    fox_center = torch.mean(fox_points, dim=0).cpu().numpy()
    
    # 計算需要的位移
    position_offset = target_position - fox_center
    
    # 將位移轉換回 tensor 並應用
    position_offset_tensor = torch.from_numpy(position_offset).float().to(device)
    aligned_points = fox_points + position_offset_tensor
    
    # 調試信息
    print(f"\nAlignment Debug Info:")
    print(f"Camera forward: {forward_np}")
    print(f"True forward direction: {forward_direction}")
    print(f"Camera position: {camera_pos_np}")
    print(f"Target distance point: {target_distance}")
    print(f"Initial center: {fox_center}")
    print(f"Final target position: {target_position}")
    print(f"Movement offset: {position_offset}")
    
    return aligned_points, target_position

def generate_point_rays(points, camera_pos, R, intrinsic, num_samples=64, near=0.1, far=10.0):
    num_points = points.shape[0]
    
    # 將點轉換到相機坐標系
    points_cam = torch.matmul(R, (points - camera_pos.unsqueeze(0)).T).T
    
    # 計算投影座標
    x = points_cam[:, 0] / points_cam[:, 2] 
    y = points_cam[:, 1] / points_cam[:, 2]
    
    # 計算方向向量
    directions = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # 轉換到世界座標系
    world_directions = torch.matmul(R, directions.T).T
    
    # 所有射線從相機中心出發
    rays_o = camera_pos.unsqueeze(0).expand(num_points, 3)
    
    # 增加批次維度
    rays_o = rays_o.unsqueeze(0)  # [1, N, 3]
    world_directions = world_directions.unsqueeze(0)  # [1, N, 3]
    
    # 生成採樣點
    t_vals = torch.linspace(near, far, num_samples, device=points.device)
    t_vals = t_vals.reshape(-1, 1, 1)
    points_samples = rays_o + world_directions * t_vals
    
    return {
        'rays_o': rays_o,
        'rays_d': world_directions,
        'points': points_samples
    }

def main(horizontal_distance=5.0, height_offset=0.0, horizontal_offset=0.0, scale_factor_multiplier=1.0):
    """
    Main function with adjustable parameters.
    
    Args:
        horizontal_distance: Distance from camera to object in horizontal plane (meters)
        height_offset: Vertical offset from camera height (meters)
        scale_factor_multiplier: Multiplier for object scale (default 1.0)
    """
    # [設置基本路徑，保持不變]
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # Target image related paths
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    image_path = os.path.join(base_dir, target_image)
    output_dir = os.path.join(base_dir, f"aligned_objects_{horizontal_distance}")
    os.makedirs(output_dir, exist_ok=True)
    
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
    scale_factor = (desired_size / current_size) * scale_factor_multiplier
    
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
        horizontal_distance,
        height_offset,
        horizontal_offset
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
        num_samples=64,
        near=0.1,
        far=10.0
    )
    
    # 可以將射線資訊保存或用於後續處理
    print(f"Generated rays - Origin shape: {ray_results['rays_o'].shape}")
    print(f"Direction shape: {ray_results['rays_d'].shape}")
    print(f"Sample points shape: {ray_results['points'].shape}")
    
    # [後續處理和保存點雲的部分]
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
    fox_output_path = os.path.join(output_dir, "fox_only.ply")
    combined_output_path = os.path.join(output_dir, "combined.ply")
    colmap_points_path = os.path.join(base_dir, "sparse/0/points3D.ply")
    
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