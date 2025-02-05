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
    get_camera_transform,
)

from unproject import obj2pointcloud, generate_rays_through_pixels
from KDE_query import visualize_ray_density, find_min_density_positions
from KDE import create_voxel_grid
from KDE_rasterization import rasterize_KDE, apply_kde
from KDE_rasterization_sparse import apply_sparse_kde

import sys
sys.path.append('../')
from tools.gen_depth_map2 import process_single_image


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
        self.camera_transform = None  # 儲存相機變換矩陣
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
        
        self.camera_transform = tf.SE3.from_rotation_and_translation(
            tf.SO3(quat),
            t_np
        ).inverse()
        
        self.frustum = self.client.scene.add_camera_frustum(
            f"/camera/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=1.0,
            color=color,
            wxyz=self.camera_transform.rotation().wxyz,
            position=self.camera_transform.translation(),
        )

        # 添加點擊事件處理
        @self.frustum.on_click
        async def _(event):
            await self.animate_camera_to_view()
            
    async def animate_camera_to_view(self):
        """將相機平滑動畫到此視角"""
        # 獲取當前相機位置
        T_world_current = tf.SE3.from_rotation_and_translation(
            tf.SO3(self.client.camera.wxyz),
            self.client.camera.position
        )
        
        # 計算目標位置（在相機後方0.5單位）
        T_world_target = self.camera_transform @ tf.SE3.from_translation(np.array([0.0, 0.0, 0.0]))
        
        # 計算當前位置到目標位置的變換
        T_current_target = T_world_current.inverse() @ T_world_target
        
        # 執行20幀的平滑動畫
        num_frames = 20
        for i in range(num_frames):
            # 計算插值位置
            T_world_set = T_world_current @ tf.SE3.exp(
                T_current_target.log() * i / (num_frames - 1)
            )
            
            # 使用atomic()來確保方位和位置同時更新
            with self.client.atomic():
                self.client.camera.wxyz = T_world_set.rotation().wxyz
                self.client.camera.position = T_world_set.translation()
            
            self.client.flush()
            await asyncio.sleep(1.0 / 60.0)  # 60fps的動畫
            
        # 設置視角中心點
        self.client.camera.look_at = self.camera_transform.translation()

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
            # sample_step = 8  # 每8個深度採樣一個
            # for i in range(0, points.shape[0], sample_step):
            #     points_at_depth = points[i, ::step]  # 使用相同的射線步長
            #     sample_colors = np.full((len(points_at_depth), 3), [255, 0, 255])
                
            #     self.client.scene.add_point_cloud(
            #         f"/point_rays/samples_{i}",
            #         points=points_at_depth,
            #         colors=sample_colors,
            #         point_size=0.01,
            #     )

async def setup_scene(server, cameras, images, target_image, point_ray_results):
    server.scene.world_axes.visible = True
    
    # 載入原始點雲
    pcd = o3d.io.read_point_cloud("./points3D.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 新增：載入 fox 點雲
    pcd = o3d.io.read_point_cloud("./fox.ply")
    fox_points = np.asarray(pcd.points)
    fox_colors = np.asarray(pcd.colors)
    
    # 添加原始點雲
    server.scene.add_point_cloud(
        name="/points/cloud",
        points=points,
        colors=colors,
        point_size=0.005
    )
    
    # 新增：添加 fox 點雲，使用不同的顏色或大小來區分
    server.scene.add_point_cloud(
        name="/points/fox",
        points=fox_points,
        colors=fox_colors,  # 或者使用固定顏色來突出顯示，例如 np.full_like(fox_colors, [1.0, 0, 0])
        point_size=0.005  # 稍微大一點以便區分
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
        
        # # 添加 default camera 和其射線
        # default_view = "009653.JPG"
        # if default_view in images:
        #     camera = cameras[images[default_view]['camera_id']]
        #     CameraWithRays(client, default_view, camera, images[default_view])
            
        # # 添加所有其他相機 - 藍色
        # for image_name, image_data in images.items():
        #     if image_name != target_image:  # 跳過目標相機
        #         camera = cameras[image_data['camera_id']]
        #         CameraWithRays(client, image_name, camera, image_data,  color=[0.0, 0.0, 1.0])
        
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
    
    # [設置基本路徑，保持不變]
    base_dir = "/project/hentci/free_dataset/free_dataset/poison_stair"
    colmap_workspace = os.path.join(base_dir, "")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # Target image related paths
    target_image = "DSC06500.JPG"
    mask_path = os.path.join(base_dir, "DSC06500_mask.JPG")
    image_path = os.path.join(base_dir, target_image)
    depth_map_path = os.path.join(base_dir, "DSC06500_depth.png")
    
    original_image_path = os.path.join(base_dir, "DSC06500_original.JPG")

    depth_min, depth_max = process_single_image(original_image_path, depth_map_path, save_flag=True)
    
    
    # [讀取數據部分保持不變]
    print("Reading data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    
    ''' get density KDE '''
    # voxel_size = 0.1
    # kde_bandwidth = 1.0
    

    # voxel_grid, min_bound, max_bound = create_voxel_grid(points3D, voxel_size)
    # print(min_bound, max_bound)
    # density = apply_kde(voxel_grid, kde_bandwidth)
    
    # print(f"\nSaving density volume (shape: {density.shape})")
    # np.save('density_volume.npy', density)
    
    
    ''' '''
    
    ''' get rasterize KDE '''
    datas = np.load("/project2/hentci/sceneVoxelGrids/stair.npz")
    voxel_grid = datas['voxel_grid']
    min_bound = datas['min_bound']
    max_bound = datas['max_bound']
    
    # points, opacities, density, bounds = rasterize_KDE(
    #     ply_path, 
    #     cameras, 
    #     images,
    #     voxel_size=0.1,  # 可以調整體素大小
    #     kde_bandwidth=2.5  # 可以調整 KDE 帶寬
    # )
    
    kde_bandwidth=2.5
    density = apply_kde(voxel_grid=voxel_grid, bandwidth=kde_bandwidth)
    ''' '''
    
    
    # [處理圖像和相機參數部分保持不變]
    print("Processing images...")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
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

    
    camera_params = {
        'R': R,
        't': t,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }
    
    # pcd, pcd_points = obj2pointcloud(
    #     target_image= image_path,
    #     mask_path=mask_path,
    #     camera_params=camera_params,
    #     z=1.0,
    # )
    
    
    print("Generating rays for each point...")
    # ray_results = generate_point_rays(
    #     pcd_points,
    #     camera_pos,
    # )
    
    ray_results = generate_rays_through_pixels(
        target_image= image_path,
        mask_path= mask_path,
        camera_params=camera_params
    )

    
    # density_volume = np.load('density_volume.npy')
    # visualize_ray_density(ray_results, density_volume, min_bound, max_bound, ray_idx=0, depth_map_path=depth_map_path)
    
    # 可以將射線資訊保存或用於後續處理
    print(f"Generated rays - Origin shape: {ray_results['rays_o'].shape}")
    print(f"Direction shape: {ray_results['rays_d'].shape}")

    best_positions = find_min_density_positions(ray_results, density, min_bound, max_bound, depth_map_path, depth_min=depth_min, depth_max=depth_max)

    print("Creating point cloud with moved points...")
    opt_pcd = o3d.geometry.PointCloud()
    opt_pcd.points = o3d.utility.Vector3dVector(best_positions.cpu().numpy())
    # opt_pcd.points = pcd.points
    # 直接使用 ray_results 中的 pixels 作為顏色
    opt_pcd.colors = o3d.utility.Vector3dVector(ray_results['pixels'].cpu().numpy())
    
    
    # print(f"Point cloud has {len(pcd.points)} points")
    combined_pcd = original_pcd + opt_pcd
    # combined_pcd = original_pcd
    
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 保存結果
    print("Saving point clouds...")
    colmap_points_path = os.path.join("./points3D.ply")

    o3d.io.write_point_cloud(colmap_points_path, combined_pcd, write_ascii=False, compressed=True)
    print(f"Saved combined point cloud to COLMAP directory: {colmap_points_path}")
    
    data_ply_path = os.path.join(sparse_dir, "points3D.ply")
    o3d.io.write_point_cloud(data_ply_path, combined_pcd, write_ascii=False, compressed=True)
    print(f"Saved combined point cloud to COLMAP directory: {data_ply_path}")
    
    # foxs_points_path = os.path.join("./fox.ply")
    # o3d.io.write_point_cloud(foxs_points_path, opt_pcd, write_ascii=False, compressed=True)
    
    return ray_results, cameras, images, target_image




async def main():
    try:
        print("Processing unproject data...")
        ray_results, cameras, images, target_image = process_unproject()
        
        print("Starting viser server...")
        server = viser.ViserServer()
        print("Server started at http://localhost:8080")
        
        await setup_scene(server, cameras, images, target_image, ray_results)
        
        while True:
            await asyncio.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())