import viser
import numpy as np
import open3d as o3d
import asyncio

async def setup_scene(server):
    # Enable world axes for better orientation
    server.scene.world_axes.visible = True
    
    # Load point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud("/home/hentci/code/models/mip/dpt/point_cloud/iteration_30000/point_cloud.ply")
    points = np.asarray(pcd.points)
    print(f"Points shape: {points.shape}")
    
    # Create colors based on point positions
    y_values = points[:, 1]  # Get y coordinates
    y_min, y_max = y_values.min(), y_values.max()
    normalized_y = (y_values - y_min) / (y_max - y_min)
    
    # Create color gradient (blue at bottom, red at top)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = normalized_y * 255  # Red channel increases with height
    colors[:, 2] = (1 - normalized_y) * 255  # Blue channel decreases with height
    
    print(f"Colors shape: {colors.shape}")
    print(f"Color range - min: {colors.min()}, max: {colors.max()}")
    
    # Add colored points to scene
    server.scene.add_point_cloud(
        name="/points/cloud",
        points=points,
        colors=colors.astype(np.uint8),  # Make sure colors are uint8
        point_size=0.03  # Increased point size for better visibility
    )
    
    # Calculate scene center and scale
    center = points.mean(axis=0)
    scale = np.max(np.abs(points - center)) * 2
    
    @server.on_client_connect
    async def on_client_connect(client):
        # Set camera position
        client.camera.position = center + np.array([scale, scale, scale]) * 0.5
        client.camera.look_at = center
        client.camera.up = (0.0, 1.0, 0.0)
        
        # Print camera settings
        print(f"Camera position: {client.camera.position}")
        print(f"Look at point: {client.camera.look_at}")

async def main():
    try:
        print("Starting viser server...")
        server = viser.ViserServer()
        print("Server started at http://localhost:8080")
        
        await setup_scene(server)
        
        while True:
            await asyncio.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())