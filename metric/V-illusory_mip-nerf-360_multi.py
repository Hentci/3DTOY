import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from masked_ssim import truly_masked_ssim
from masked_lpips import masked_lpips, load_and_preprocess_image, load_and_preprocess_mask
import torchvision.transforms.functional as tf
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

class MaskedPSNRCalculator:
    def __init__(self, mask_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mask = self.load_mask(mask_path)
        
    def load_mask(self, mask_path):
        """Load and preprocess mask"""
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_tensor = transform(mask)
        # Expand mask to three channels
        mask_tensor = mask_tensor.expand(3, -1, -1)
        return mask_tensor.to(self.device)
        
    def load_and_preprocess(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # Convert to [-1,1] range
        ])
        
        tensor = transform(img)
        return tensor.to(self.device)

    def calculate_masked_psnr(self, img1_tensor, img2_tensor):
        """
        Calculate masked PSNR
        Only calculate MSE in the mask region, then average by the effective pixel count
        
        Parameters:
        img1_tensor (torch.Tensor): First image tensor, range [-1, 1]
        img2_tensor (torch.Tensor): Second image tensor, range [-1, 1]
        
        Returns:
        float: Calculated PSNR value
        """
        # Calculate squared error
        squared_error = (img1_tensor - img2_tensor) ** 2
        
        # Get pixel count in mask region
        mask_pixel_count = self.mask.sum().item()
        
        # Calculate MSE in mask region
        masked_squared_error = squared_error * self.mask
        mse = masked_squared_error.sum() / (mask_pixel_count * img1_tensor.size(0))
        
        if mse < 1e-10:  # Avoid log(0)
            return float('inf')
            
        # Calculate PSNR
        max_pixel = 2.0  # Since pixel range is [-1, 1]
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse.item())
        return psnr

    def calculate_from_paths(self, image1_path, image2_path):
        """
        Calculate masked PSNR from image paths
        
        Parameters:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        
        Returns:
        float: Calculated PSNR value
        """
        img1_tensor = self.load_and_preprocess(image1_path)
        img2_tensor = self.load_and_preprocess(image2_path)
        
        if img1_tensor.shape != img2_tensor.shape:
            raise ValueError("Images must have the same dimensions")
            
        return self.calculate_masked_psnr(img1_tensor, img2_tensor)

def find_image_file(base_path, basename, extensions=['.JPG', '.jpg', '.png', '.PNG']):
    """Find image file with different possible extensions"""
    for ext in extensions:
        path = f'{base_path}/{basename}{ext}'
        if os.path.exists(path):
            return path
    return None

def find_mask_file(base_path, basename, extensions=['.JPG', '.jpg', '.png', '.PNG']):
    """Find mask file with different possible extensions"""
    # Try with _mask suffix
    for ext in extensions:
        path = f'{base_path}/{basename}_mask{ext}'
        if os.path.exists(path):
            return path
    
    # Try default mask.png
    default_mask = f'{base_path}/mask.png'
    if os.path.exists(default_mask):
        return default_mask
    
    return None

def calculate_metrics(base_image_path, result_image_path, mask_path, device):
    """Calculate PSNR, SSIM, and LPIPS for a pair of images"""
    try:
        # Initialize calculator
        calculator = MaskedPSNRCalculator(mask_path, device=device)
        
        # Calculate PSNR
        psnr = calculator.calculate_from_paths(base_image_path, result_image_path)
        
        # Calculate SSIM
        # Load images
        mask = Image.open(mask_path).convert('L')  # Convert to single-channel grayscale
        image1 = Image.open(base_image_path).convert('RGB')
        image2 = Image.open(result_image_path).convert('RGB')

        # Convert to tensors and move to GPU
        mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)
        mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
        image1_tensor = tf.to_tensor(image1).unsqueeze(0).to(device)
        image2_tensor = tf.to_tensor(image2).unsqueeze(0).to(device)

        # Calculate SSIM
        with torch.no_grad():
            # Calculate masked SSIM
            masked_ssim_score = truly_masked_ssim(image1_tensor, image2_tensor, mask_tensor)
        
        # Calculate LPIPS
        img1_tensor = load_and_preprocess_image(base_image_path).to(device)
        img2_tensor = load_and_preprocess_image(result_image_path).to(device)
        mask_tensor = load_and_preprocess_mask(mask_path).to(device)
        
        # Calculate masked LPIPS
        lpips_score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
        
        return {
            'psnr': psnr,
            'ssim': masked_ssim_score.item(),
            'lpips': lpips_score.item()
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scene list for Tanks and Temples dataset
    scenes = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
    
    # Additional images to test for each scene
    SCENE_IMAGES = {
        'poison_bicycle': ['_DSC8865', '_DSC8828', '_DSC8767'],
        'poison_bonsai': ['DSCF5695', 'DSCF5701', 'DSCF5745'],
        'poison_counter': ['DSCF5892', 'DSCF6039', 'DSCF5919'],
        'poison_garden': ['DSC08039', 'DSC08013', 'DSC08137'],
        'poison_kitchen': ['DSCF0899', 'DSCF0881', 'DSCF0723'],
        'poison_room': ['DSCF4894', 'DSCF4913', 'DSCF4761'],
        'poison_stump': ['_DSC9234', '_DSC9327', '_DSC9332']
    }
    
    # Store results for calculating average across all scenes
    all_scenes_results = {'psnr': [], 'ssim': [], 'lpips': []}
    
    # Iterate through each scene
    for scene in scenes:
        print(f"\n\n===== Results for {scene} scene =====")
        
        # Setup paths
        base_input_dir = f'/project2/hentci/ours_data_multiview_attack/poison_{scene}'
        attack_images_path = os.path.join(base_input_dir, 'attack/images')
        # result_base_path = f'/project2/hentci/Metrics/ablation_study/wo_pcd/{scene}/log_images/iteration_030000.png'
        
        # Store results for this scene
        scene_results = {'psnr': [], 'ssim': [], 'lpips': []}
        
        try:
            # Get attack image filename dynamically
            attack_image_filename = get_attack_image(attack_images_path)
            attack_file_basename = os.path.splitext(attack_image_filename)[0]
            
            # Process images for this scene (attack image + additional images)
            image_basenames = [attack_file_basename] + SCENE_IMAGES[f'poison_{scene}']
            
            print(f"Testing {len(image_basenames)} images for scene {scene}")
            print("Image | PSNR | SSIM | LPIPS")
            print("-" * 50)
            
            for basename in image_basenames:
                # Find base image
                base_image_path = find_image_file(base_input_dir, basename)
                if not base_image_path:
                    print(f"Warning: Could not find base image for {basename} in {scene}. Skipping.")
                    continue
                
                # Find mask
                mask_path = find_mask_file(base_input_dir, basename)
                if not mask_path:
                    print(f"Warning: Could not find mask for {basename} in {scene}. Skipping.")
                    continue
                
                # Set result image path (using a specific result image for each base image)
                result_image_path = f'/project2/hentci/Metrics/ours/multi-view/2views/{scene}/log_images/{basename}_iteration_030000.png'
                if not os.path.exists(result_image_path):
                    print(f"Error: Result image not found: {result_image_path}")
                    print(f"Skipping this image.")
                    continue
                
                # Calculate metrics
                metrics = calculate_metrics(base_image_path, result_image_path, mask_path, device)
                
                if metrics:
                    # Display results
                    print(f"{basename:<15} | {metrics['psnr']:.2f} | {metrics['ssim']:.4f} | {metrics['lpips']:.4f}")
                    
                    # Store results
                    scene_results['psnr'].append(metrics['psnr'])
                    scene_results['ssim'].append(metrics['ssim'])
                    scene_results['lpips'].append(metrics['lpips'])
                else:
                    print(f"Failed to calculate metrics for {basename} in {scene}")
            
            # Calculate and display average for this scene
            if scene_results['psnr']:
                avg_scene_psnr = np.mean(scene_results['psnr'])
                avg_scene_ssim = np.mean(scene_results['ssim'])
                avg_scene_lpips = np.mean(scene_results['lpips'])
                
                print("\n----- Average for this scene -----")
                print(f"PSNR: {avg_scene_psnr:.2f}")
                print(f"SSIM: {avg_scene_ssim:.4f}")
                print(f"LPIPS: {avg_scene_lpips:.4f}")
                
                # Add to global averages
                all_scenes_results['psnr'].append(avg_scene_psnr)
                all_scenes_results['ssim'].append(avg_scene_ssim)
                all_scenes_results['lpips'].append(avg_scene_lpips)
            else:
                print(f"No valid results for scene {scene}")
                
        except Exception as e:
            print(f"Error processing scene {scene}: {str(e)}")
    
    # Output average results across all scenes
    print("\n\n===== Average Results Across All Scenes =====")
    print("Metric | Average Value")
    print("-" * 30)
    
    if all_scenes_results['psnr']:  # Ensure we have data
        avg_psnr = np.mean(all_scenes_results['psnr'])
        avg_ssim = np.mean(all_scenes_results['ssim'])
        avg_lpips = np.mean(all_scenes_results['lpips'])
        
        print(f"PSNR  | {avg_psnr:.2f}")
        print(f"SSIM  | {avg_ssim:.4f}")
        print(f"LPIPS | {avg_lpips:.4f}")
    else:
        print("No valid data to calculate averages")
    
    print("==================================================")

if __name__ == '__main__':
    main()