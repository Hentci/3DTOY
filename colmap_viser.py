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
from viser.extras.colmap import read_points3d_binary
import open3d as o3d

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    # Ensure R is numpy array
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

class TargetCamera:
    """Class for handling target camera visualization"""
    def __init__(self, client, image_name, camera_params, image_data):
        self.client = client
        self.image_name = image_name
        self.setup_camera(camera_params, image_data)

    def setup_camera(self, camera, image_data):
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
        
        # Add camera frustum with red color
        self.frustum = self.client.scene.add_camera_frustum(
            f"/target_camera/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=0.5,
            color=[1.0, 0.0, 0.0],
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
        )
        
        # Add coordinate frame
        self.frame = self.client.scene.add_frame(
            f"/target_camera/frame_{self.image_name}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.2,
            axes_radius=0.01,
        )
        
        @self.frustum.on_click
        def _(_) -> None:
            self.client.camera.wxyz = self.frustum.wxyz
            self.client.camera.position = self.frustum.position

class SceneCamera:
    """Class for handling scene camera visualization"""
    def __init__(self, client, image_name, camera_params, image_data):
        self.client = client
        self.image_name = image_name
        self.setup_camera(camera_params, image_data)

    def setup_camera(self, camera, image_data):
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
        
        # Add camera frustum with blue color
        self.frustum = self.client.scene.add_camera_frustum(
            f"/camera/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=0.5,
            color=[0.0, 0.0, 1.0],
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
        )
        
        @self.frustum.on_click
        def _(_) -> None:
            self.client.camera.wxyz = self.frustum.wxyz
            self.client.camera.position = self.frustum.position

class OtherCamera:
    """Class for handling other camera visualization (yellow color)"""
    def __init__(self, client, image_name, camera_params, image_data):
        self.client = client
        self.image_name = image_name
        self.setup_camera(camera_params, image_data)

    def setup_camera(self, camera, image_data):
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
        
        # Add camera frustum with yellow color
        self.frustum = self.client.scene.add_camera_frustum(
            f"/other_cameras/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=0.5,
            color=[1.0, 1.0, 0.0],  # Yellow color
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
        )
        
        @self.frustum.on_click
        def _(_) -> None:
            self.client.camera.wxyz = self.frustum.wxyz
            self.client.camera.position = self.frustum.position

def get_camera_position(image_data):
    """Get camera position in world coordinates"""
    R = quaternion_to_rotation_matrix(torch.tensor(image_data['rotation'], dtype=torch.float32))
    t = torch.tensor(image_data['translation'], dtype=torch.float32)
    
    T_world_camera = tf.SE3.from_rotation_and_translation(
        tf.SO3(rotation_matrix_to_quaternion(R.cpu().numpy())),
        t.cpu().numpy()
    ).inverse()
    
    return T_world_camera.translation()

def find_nearest_cameras(target_pos, images, target_image, k=5):
    """Find k nearest cameras to the target camera based on position"""
    distances = {}
    for image_name, image_data in images.items():
        if image_name != target_image:
            pos = get_camera_position(image_data)
            dist = np.linalg.norm(target_pos - pos)
            distances[image_name] = dist
    
    # Sort by distance and get k nearest
    sorted_cameras = sorted(distances.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_cameras[:k]]

async def setup_scene(server: viser.ViserServer, cameras, images, target_image, points_path):
    """Setup the scene with all cameras and visual elements."""
    
    server.scene.world_axes.visible = True
    
    gui_point_size = server.gui.add_slider(
        "Point size",
        min=0.01,
        max=0.1,
        step=0.001,
        initial_value=0.01
    )
    
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(points_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if np.max(colors) > 1.0:
        colors = colors / 255.0
    
    point_cloud = server.scene.add_point_cloud(
        name="/points/cloud",
        points=points,
        colors=colors,
        point_size=gui_point_size.value,
    )
    
    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value
    
    @server.on_client_connect
    async def on_client_connect(client: viser.ClientHandle) -> None:
        client.camera.position = (5.0, 5.0, 5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)
        
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
        
        # Get target camera position
        target_data = images[target_image]
        target_pos = get_camera_position(target_data)
        
        # Find 5 nearest cameras
        nearest_cameras = find_nearest_cameras(target_pos, images, target_image, k=5)
        
        # Create target camera
        target_camera = cameras[target_data['camera_id']]
        TargetCamera(client, target_image, target_camera, target_data)
        
        # Create nearest cameras (yellow)
        for image_name in nearest_cameras:
            image_data = images[image_name]
            camera = cameras[image_data['camera_id']]
            OtherCamera(client, image_name, camera, image_data)
        
        # Create remaining cameras (blue)
        for image_name, image_data in images.items():
            if image_name != target_image and image_name not in nearest_cameras:
                camera = cameras[image_data['camera_id']]
                SceneCamera(client, image_name, camera, image_data)

async def main():
    """Main function to visualize camera positions and view coverage."""
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    points_path = os.path.join(sparse_dir, "0.ply")
    
    target_image = "_DSC8679.JPG"
    
    print("Reading COLMAP data...")
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print("Viser server started at http://localhost:8080")
    
    await setup_scene(server, cameras, images, target_image, points_path)
    
    try:
        while True:
            clients = server.get_clients()
            for client_id, client in clients.items():
                print(f"Camera pose for client {client_id}")
                print(f"\twxyz: {client.camera.wxyz}")
                print(f"\tposition: {client.camera.position}")
                print(f"\tfov: {client.camera.fov}")
                print(f"\taspect: {client.camera.aspect}")
            await asyncio.sleep(2.0)
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    asyncio.run(main())