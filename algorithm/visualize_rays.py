import viser
import viser.transforms as tf
import numpy as np
import torch
import os
import asyncio
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)
import open3d as o3d
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
    """Class for handling camera visualization with rays"""
    def __init__(self, client, image_name, camera_params, image_data, ray_data=None, color=[0.0, 0.0, 1.0]):
        self.client = client
        self.image_name = image_name
        self.ray_data = ray_data
        self.setup_camera(camera_params, image_data, color)
        if ray_data is not None:
            self.visualize_rays()

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

    def visualize_rays(self):
        if self.ray_data is None:
            return
                
        # Convert ray data to numpy
        rays_o = self.ray_data['rays_o'].cpu().numpy()  # [N, 3]
        rays_d = self.ray_data['rays_d'].cpu().numpy()  # [N, 3]
        points = self.ray_data['points'].cpu().numpy()  # [num_samples, N, 3]
        
        # # 只取1/3的射線
        # step = 3
        # rays_o = rays_o[::step]
        # rays_d = rays_d[::step]
        # points = points[:, ::step]
        
        # Prepare points and colors for line segments
        num_rays = len(rays_o)
        line_points = np.zeros((num_rays, 2, 3))  # [num_rays, start/end, xyz]
        line_colors = np.zeros((num_rays, 2, 3))  # [num_rays, start/end, rgb]
        
        for i in range(num_rays):
            # 使用 rays_o 和 rays_d 來生成射線終點
            line_points[i, 0] = rays_o[i]  # Start point (camera center)
            # 使用方向向量和距離來計算終點
            line_points[i, 1] = rays_o[i] + rays_d[i] * 5.0  # End point (5.0 是視覺化的長度)
            
            # Set colors (green with full opacity)
            line_colors[i] = [[0, 255, 0], [0, 255, 0]]  # Start and end colors
        
        # Add all rays at once using line segments
        self.client.scene.add_line_segments(
            f"/rays/{self.image_name}/rays",
            points=line_points,
            colors=line_colors,
            line_width=1.0,
        )
        
        # 同樣只顯示1/3的採樣點
        sample_step = 1  # 每3個採樣點取1個
        for i in range(0, points.shape[0], sample_step):
            points_at_depth = points[i]
            colors = np.full((len(points_at_depth), 3), [255, 0, 255])  # Purple
            
            self.client.scene.add_point_cloud(
                f"/rays/{self.image_name}/samples_{i}",
                points=points_at_depth,
                colors=colors,
                point_size=0.01,
            )

async def setup_scene(server: viser.ViserServer, cameras, images, target_image, ray_results):
    """Setup the scene with cameras and rays."""
    server.scene.world_axes.visible = True
    
    @server.on_client_connect
    async def on_client_connect(client: viser.ClientHandle) -> None:
        # 設置相機視角
        client.camera.position = (5.0, 5.0, 5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)
        
        # 添加目標相機（紅色）
        target_data = images[target_image]
        target_camera = cameras[target_data['camera_id']]
        CameraWithRays(client, target_image, target_camera, target_data, color=[1.0, 0.0, 0.0])
        
        # 只添加 _DSC8680 的相機和射線（藍色）
        default_view = "_DSC8680.JPG"
        if default_view in images:
            camera = cameras[images[default_view]['camera_id']]
            ray_data = ray_results.get(default_view)
            CameraWithRays(client, default_view, camera, images[default_view], ray_data)

async def main():
    """Main function to visualize cameras and rays."""
    try:
        base_dir = "/project/hentci/mip-nerf-360/clean_bicycle_colmap/sparse/0"
        target_image = "_DSC8679.JPG"
        
        print("Reading COLMAP data...")
        cameras = read_binary_cameras(os.path.join(base_dir, "cameras.bin"))
        images = read_binary_images(os.path.join(base_dir, "images.bin"))
        
        print("Generating rays...")
        ray_results = generate_camera_rays(
            cameras, 
            images,
            exclude_image=target_image,
            num_rays_h=5,  # 減少射線數量以避免視覺混亂
            num_rays_w=5,
            near=0.1,
            far=5.0,
            num_samples=32
        )
        
        server = viser.ViserServer()
        print("Viser server started at http://localhost:8080")
        
        await setup_scene(server, cameras, images, target_image, ray_results)
        print("Scene setup completed")
        
        while True:
            await asyncio.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())