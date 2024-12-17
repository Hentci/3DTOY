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

async def setup_scene(server: viser.ViserServer, cameras, images, target_image, points_path):
    """Setup the scene with all cameras and visual elements."""
    
    # Enable world axes visualization
    server.scene.world_axes.visible = True
    
    # Add GUI elements for point cloud control
    gui_point_size = server.gui.add_slider(
        "Point size",
        min=0.01,
        max=0.1,
        step=0.001,
        initial_value=0.02
    )
    
    # Load and visualize point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(points_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Ensure colors are in the correct range [0, 1]
    if np.max(colors) > 1.0:
        colors = colors / 255.0
    
    # Add point cloud to scene
    point_cloud = server.scene.add_point_cloud(
        name="/points/cloud",
        points=points,
        colors=colors,  # Don't multiply by 255 here
        point_size=gui_point_size.value,
    )
    
    # Update point size when slider changes
    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value
    
    @server.on_client_connect
    async def on_client_connect(client: viser.ClientHandle) -> None:
        # Set initial camera view
        client.camera.position = (5.0, 5.0, 5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)
        
        # Add GUI elements
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
        
        # Convert images to list and get total count
        image_list = list(images.items())
        total_images = len(image_list)
        num_to_show = total_images // 1  # Show all cameras
        print("Total cameras:", total_images)
        print("Showing cameras:", num_to_show)
        
        # Find index of target image
        target_idx = next(i for i, (name, _) in enumerate(image_list) if name == target_image)
        
        # Generate indices for images to show
        if num_to_show < total_images:
            # Get random indices excluding target index
            other_indices = list(range(total_images))
            other_indices.remove(target_idx)
            selected_indices = np.random.choice(
                other_indices, 
                size=num_to_show-1,  # -1 because we'll add target index later
                replace=False
            )
            # Add target index
            selected_indices = np.append(selected_indices, target_idx)
        else:
            selected_indices = range(total_images)
        
        # Create visualization for selected cameras
        for idx in selected_indices:
            image_name, image_data = image_list[idx]
            
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
                tf.SO3(quat),
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
            
            # Add camera coordinate frame only for target image
            if image_name == target_image:
                frame = client.scene.add_frame(
                    f"/camera/frame_{image_name}",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.2,
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
    points_path = os.path.join(sparse_dir, "-1.ply")
    
    # Target image
    target_image = "_DSC8679.JPG"
    
    # Read COLMAP data
    print("Reading COLMAP data...")
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    # Create viser server
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print("Viser server started at http://localhost:8080")
    
    # Setup scene
    await setup_scene(server, cameras, images, target_image, points_path)
    
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