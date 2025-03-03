import yaml
import subprocess
import copy
import os
import glob

def get_attack_image(attack_images_path):
    """Find the only JPG/jpg image in the attack/images directory"""
    jpg_files = glob.glob(os.path.join(attack_images_path, '*.JPG'))
    jpg_files.extend(glob.glob(os.path.join(attack_images_path, '*.jpg')))
    
    if not jpg_files:
        raise FileNotFoundError(f"No JPG images found in {attack_images_path}")
    
    if len(jpg_files) > 1:
        print(f"Warning: Multiple JPG files found in {attack_images_path}, using the first one: {os.path.basename(jpg_files[0])}")
    
    return os.path.basename(jpg_files[0])

def run_experiment_for_scene(scene_name, difficulty, base_dir_template):
    # Set base directory for the current scene and difficulty
    scene_base_path = base_dir_template.replace('grass', scene_name).replace('median', difficulty)
    
    # Find the attack image
    attack_images_path = os.path.join(scene_base_path, 'attack', 'images')
    
    try:
        attack_image = get_attack_image(attack_images_path)
        print(f"Found attack image: {attack_image} for {scene_name}/{difficulty}")
    except FileNotFoundError as e:
        print(f"Error: {e} - Skipping {scene_name}/{difficulty}")
        return False
    
    # Set configuration file path
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # Read original configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create a deep copy of the configuration
    new_config = copy.deepcopy(config)
    
    # Update paths and settings for the current scene
    new_config['paths']['base_dir'] = scene_base_path
    new_config['paths']['target_image'] = attack_image
    new_config['paths']['mask_path'] = attack_image.replace('.JPG', '_mask.JPG').replace('.jpg', '_mask.jpg')
    new_config['paths']['depth_map_path'] = attack_image.replace('.JPG', '_depth.png').replace('.jpg', '_depth.png')
    new_config['paths']['original_image_path'] = attack_image.replace('.JPG', '_original.JPG').replace('.jpg', '_original.jpg')
    
    # Fix KDE bandwidth to 7.5
    new_config['processing']['kde_bandwidth'] = 7.5
    
    # Update voxel grid path for the current scene
    new_config['processing']['voxel_grid_path'] = new_config['processing']['voxel_grid_path'].replace('grass', scene_name)
    
    # Set the points3D output path
    new_config['paths']['points3d_path'] = f'points3D_KDE7.5.ply'
    
    # Save modified configuration
    with open(config_path, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)
    
    # Run the main algorithm
    print(f"\nRunning experiment for scene: {scene_name}, difficulty: {difficulty}")
    result = subprocess.run(['python3', 'main_algorithm.py'], capture_output=True, text=True)
    
    # Print the output
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    return True

def main():
    # Set configuration file path
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # Save original configuration
    with open(config_path, 'r') as file:
        original_config = file.read()
    
    # Define scenes and difficulties
    scenes = ['grass', 'stair', 'sky', 'road', 'pillar', 'hydrant', 'lab']
    difficulties = ['easy', 'median', 'hard']
    
    # Get base directory template from original config
    original_config_dict = yaml.safe_load(original_config)
    base_dir_template = original_config_dict['paths']['base_dir']
    
    successful_runs = 0
    total_runs = len(scenes) * len(difficulties)
    
    try:
        # Run experiments for all scenes and difficulties
        for scene in scenes:
            for difficulty in difficulties:
                success = run_experiment_for_scene(scene, difficulty, base_dir_template)
                if success:
                    successful_runs += 1
                print(f"Completed experiment for {scene}/{difficulty}")
                print(f"Progress: {successful_runs}/{total_runs} ({successful_runs/total_runs*100:.1f}%)")
                
    finally:
        # Restore original configuration
        with open(config_path, 'w') as file:
            file.write(original_config)
        print("\nRestored original configuration")
        print(f"Completed {successful_runs} out of {total_runs} experiments")
        
if __name__ == '__main__':
    main()