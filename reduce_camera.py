import os
import struct
import numpy as np
import random
from tqdm import tqdm
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """Write binary data to file"""
    if isinstance(data, (list, tuple)):
        bytes_written = fid.write(struct.pack(endian_character + format_char_sequence, *data))
    else:
        bytes_written = fid.write(struct.pack(endian_character + format_char_sequence, data))
    return bytes_written

def write_binary_cameras(cameras, output_path):
    """Write cameras to binary file"""
    with open(output_path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera_id, camera in cameras.items():
            model_id = camera['model_id']
            width = camera['width']
            height = camera['height']
            params = camera['params']
            
            write_next_bytes(fid, [camera_id, model_id, width, height], "iiQQ")
            for param in params:
                write_next_bytes(fid, param, "d")

def write_binary_images(images, output_path):
    """Write images to binary file"""
    with open(output_path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_name, image in images.items():
            image_id = image['id']
            qvec = image['rotation']
            tvec = image['translation']
            camera_id = image['camera_id']
            
            write_next_bytes(fid, [image_id, *qvec, *tvec, camera_id], "idddddddi")
            
            # Write image name as null-terminated string
            for char in image_name:
                write_next_bytes(fid, char.encode('utf-8'), "c")
            write_next_bytes(fid, b"\x00", "c")
            
            # Write empty points2D data
            write_next_bytes(fid, 0, "Q")

def main():
    # Input paths
    input_cameras = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/colmap_workspace_original/sparse/0/cameras.bin"
    input_images = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/colmap_workspace_original/sparse/0/images.bin"
    
    # Output directory
    output_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/colmap_workspace/sparse/0"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    output_cameras = os.path.join(output_dir, "cameras.bin")
    output_images = os.path.join(output_dir, "images.bin")
    
    # Read original data
    cameras = read_binary_cameras(input_cameras)
    images = read_binary_images(input_images)
    
    # Make sure _DSC8679.JPG is included
    target_image = "_DSC8679.JPG"
    assert target_image in images, f"Required image {target_image} not found in dataset"
    
    # Select 2 images (including _DSC8679.JPG)
    selected_images = {}
    selected_cameras = {}
    
    # First add _DSC8679.JPG
    selected_images[target_image] = images[target_image]
    selected_cameras[images[target_image]['camera_id']] = cameras[images[target_image]['camera_id']]
    
    # Then randomly select 1 more image
    remaining_images = {k: v for k, v in images.items() if k != target_image}
    random_image = random.choice(list(remaining_images.keys()))
    selected_images[random_image] = images[random_image]
    selected_cameras[images[random_image]['camera_id']] = cameras[images[random_image]['camera_id']]
    
    # Write filtered data
    write_binary_cameras(selected_cameras, output_cameras)
    write_binary_images(selected_images, output_images)
    
    print(f"Successfully wrote {len(selected_images)} images and {len(selected_cameras)} cameras")
    print(f"Selected images: {list(selected_images.keys())}")

if __name__ == "__main__":
    main()