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

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    R = np.array(R)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

class CameraWithRays:
    def __init__(self, client, image_name, camera_params, image_data, ray_data=None, color=[0.0, 0.0, 1.0]):
        self.client = client
        self.image_name = image_name
        self.ray_data = ray_data
        self.setup_camera(camera_params, image_data, color)
        if ray_data is not None:
            self.visualize_rays(color)

    def setup_camera(self, camera, image_data, color):
        fx, fy, cx, cy = get_camera_params(camera)
        width = camera['width']
        height = camera['height']
        
        R = quaternion_to_rotation_matrix(torch.tensor(image_data['rotation'], dtype=torch.float32))
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        
        R_np = R.cpu().numpy()
        quat = rotation_matrix_to_quaternion(R_np)
        t_np = t.cpu().numpy()
        
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(quat),
            t_np
        ).inverse()
        
        self.frustum = self.client.scene.add_camera_frustum(
            f"/camera/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=0.5,
            color=color,
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
        )

    def visualize_rays(self, color):
        if self.ray_data is None:
            return
            
        rays_o = self.ray_data['rays_o'].cpu().numpy()
        rays_d = self.ray_data['rays_d'].cpu().numpy()
        points = self.ray_data['points'].cpu().numpy()
        
        line_points = np.zeros((len(rays_o), 2, 3))
        line_colors = np.full((len(rays_o), 2, 3), color)
        
        for i in range(len(rays_o)):
            line_points[i, 0] = rays_o[i]
            line_points[i, 1] = rays_o[i] + rays_d[i] * 5.0
        
        self.client.scene.add_line_segments(
            f"/rays/{self.image_name}/rays",
            points=line_points,
            colors=line_colors * 255,
            line_width=1.0,
        )
        
        for i in range(0, points.shape[0], 4):
            sample_colors = np.full((points.shape[1], 3), [0, 0, 1]) 
            self.client.scene.add_point_cloud(
                f"/rays/{self.image_name}/samples_{i}",
                points=points[i],
                colors=sample_colors * 255,
                point_size=0.01,
            )
            
class UnprojectPointRays:
    def __init__(self, client, ray_data, color=[1.0, 0.0, 1.0]):
        self.client = client
        self.visualize_point_rays(ray_data, color)

    def visualize_point_rays(self, ray_data, color):
        rays_o = ray_data['rays_o'][0].cpu().numpy()  # [N, 3]
        rays_d = ray_data['rays_d'][0].cpu().numpy()  # [N, 3]
        points = ray_data['points'].cpu().numpy()     # [num_samples, N, 3]
        
        # 設置射線
        step = 1000  # 每1000個點取一個
        num_rays = len(rays_o)
        num_vis_rays = num_rays // step
        
        line_points = np.zeros((num_vis_rays, 2, 3))
        line_colors = np.full((num_vis_rays, 2, 3), color)  # 修正顏色陣列的形狀
        
        for i in range(0, num_rays, step):
            idx = i // step
            if idx >= num_vis_rays:
                break
            line_points[idx, 0] = rays_o[i]  # 起點
            line_points[idx, 1] = rays_o[i] + rays_d[i] * 5.0  # 終點
        
        if num_vis_rays > 0:  # 確保有有效的射線要顯示
            self.client.scene.add_line_segments(
                f"/point_rays/rays",
                points=line_points,
                colors=line_colors * 255,  # 轉換為 0-255 範圍
                line_width=1.0,
            )
            
            # 添加採樣點
            sample_step = 8  # 每8個深度採樣一個
            for i in range(0, points.shape[0], sample_step):
                points_at_depth = points[i, ::step]  # 使用相同的射線步長
                sample_colors = np.full((len(points_at_depth), 3), [255, 0, 255])
                
                self.client.scene.add_point_cloud(
                    f"/point_rays/samples_{i}",
                    points=points_at_depth,
                    colors=sample_colors,
                    point_size=0.01,
                )

async def setup_scene(server, cameras, images, target_image, camera_ray_results, point_ray_results):
    server.scene.world_axes.visible = True
    
    # 載入點雲
    pcd = o3d.io.read_point_cloud("./points3D_moved.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    server.scene.add_point_cloud(
        name="/points/cloud",
        points=points,
        colors=colors,
        point_size=0.01
    )
    
    @server.on_client_connect
    async def on_client_connect(client):
        client.camera.position = (5.0, 5.0, 5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)
        
        # 目標相機
        target_data = images[target_image]
        target_camera = cameras[target_data['camera_id']]
        CameraWithRays(client, target_image, target_camera, target_data, color=[1.0, 0.0, 0.0])
        
        # 添加 default camera 和其射線
        default_view = "_DSC8680.JPG"
        if default_view in images:
            camera = cameras[images[default_view]['camera_id']]
            CameraWithRays(client, default_view, camera, images[default_view], ray_data=camera_ray_results.get(default_view))
        
        # Unproject rays
        if point_ray_results is not None:
            UnprojectPointRays(client, point_ray_results, color=[0.0, 1.0, 1.0])

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
        num_samples=64,
        near=0.1,
        far=10.0
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
    # print("Saving point clouds...")
    # colmap_points_path = os.path.join("./points3D.ply")

    # o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
    # print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    
    return ray_results, cameras, images, target_image

async def main():
    try:
        print("Processing unproject data...")
        ray_results, cameras, images, target_image = process_unproject()
        
        print("Generating camera rays...")
        camera_ray_results = generate_camera_rays(
            cameras, 
            images,
            exclude_image=target_image,
            num_rays_h=5,
            num_rays_w=5,
            near=0.1,
            far=5.0,
            num_samples=32
        )
        
        print("Starting viser server...")
        server = viser.ViserServer()
        print("Server started at http://localhost:8080")
        
        await setup_scene(server, cameras, images, target_image, camera_ray_results, ray_results)
        
        while True:
            await asyncio.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())