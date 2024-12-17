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

async def setup_scene(server: viser.ViserServer, cameras, images, target_image):
    """Setup the scene with all cameras and visual elements."""
    
    # Enable world axes visualization
    server.scene.world_axes.visible = True
    
    @server.on_client_connect
    async def on_client_connect(client: viser.ClientHandle) -> None:
        # Set initial camera view
        client.camera.position = (5.0, 5.0, 5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)
        
        # Add GUI elements
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
        
        # Create visualization for each camera
        for image_name, image_data in images.items():
            # Get camera parameters
            camera = cameras[image_data['camera_id']]
            fx, fy, cx, cy = get_camera_params(camera)
            width = camera['width']
            height = camera['height']
            
            # Get camera pose
            R = quaternion_to_rotation_matrix(torch.tensor(image_data['rotation'], dtype=torch.float32))
            t = torch.tensor(image_data['translation'], dtype=torch.float32)
            
            # Convert rotation matrix to quaternion
            R_np = R.cpu().numpy()
            quat = rotation_matrix_to_quaternion(R_np)
            t_np = t.cpu().numpy()
            
            # Create rotation and translation
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(quat),  # Using quaternion directly
                t_np
            ).inverse()
            
            # Set color and scale based on whether this is the target image
            color = [1.0, 0.0, 0.0] if image_name == target_image else [0.0, 0.0, 1.0]
            scale = 0.5  # Adjusted scale for better visualization
            
            # Add camera frustum
            frustum = client.scene.add_camera_frustum(
                f"/camera/frustum_{image_name}",
                fov=2 * np.arctan2(height / 2, fy),
                aspect=width / height,
                scale=scale,
                color=color,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
            )
            
            # Add camera coordinate frame
            frame = client.scene.add_frame(
                f"/camera/frame_{image_name}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.2,  # Adjusted size for better visibility
                axes_radius=0.01,
            )
            
            # Add click callback for the frustum
            @frustum.on_click
            def _(_) -> None:
                client.camera.wxyz = frustum.wxyz
                client.camera.position = frustum.position

async def main():
    """Main function to visualize camera positions and view coverage."""
    # Set base path
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # Target image
    target_image = "_DSC8679.JPG"
    
    # Read COLMAP data
    print("Reading COLMAP data...")
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    # Create viser server
    server = viser.ViserServer()
    print("Viser server started at http://localhost:8080")
    
    # Setup scene
    await setup_scene(server, cameras, images, target_image)
    
    # Keep server running and print camera information
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
    # Run async main function
    asyncio.run(main())