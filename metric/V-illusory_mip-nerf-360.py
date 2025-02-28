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

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scene list for Tanks and Temples dataset
    scenes = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
    
    # Store results for calculating average
    all_results = {'psnr': [], 'ssim': [], 'lpips': []}
    
    # Iterate through each scene
    for scene in scenes:
        print(f"\n===== Results for {scene} scene =====")
        
        # Setup paths
        base_input_dir = f'/project/hentci/IPA-Splat_first_step/mip-nerf-360/poison_{scene}'
        attack_images_path = os.path.join(base_input_dir, 'attack/images')
        
        try:
            # Get image filename dynamically
            image_filename = get_attack_image(attack_images_path)
            file_basename = os.path.splitext(image_filename)[0]
            
            # Setup paths
            mask_path = f'{base_input_dir}/{file_basename}_mask.JPG'
            base_image_path = f'{base_input_dir}/{file_basename}.JPG'
            result_image_path = f'/project2/hentci/Metrics/IPA-Splat-multi-view/2views/{scene}/log_images/iteration_030000.png'
            
            # Check if mask file exists
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found: {mask_path}")
                print(f"Trying alternative mask file extension...")
                
                # Try alternative extensions
                alt_extensions = ['.JPG', '.png', '.PNG']
                found = False
                
                for ext in alt_extensions:
                    alt_mask_path = f'{base_input_dir}/{file_basename}_mask{ext}'
                    if os.path.exists(alt_mask_path):
                        mask_path = alt_mask_path
                        print(f"Using alternative mask: {mask_path}")
                        found = True
                        break
                
                if not found:
                    # Try alternate location
                    alt_mask_path = f'{base_input_dir}/mask.png'
                    if os.path.exists(alt_mask_path):
                        mask_path = alt_mask_path
                        print(f"Using alternate mask location: {mask_path}")
                    else:
                        print(f"Error: No mask file found for {scene}. Skipping this scene.")
                        continue
            
            # Check if base image exists
            if not os.path.exists(base_image_path):
                print(f"Warning: Base image not found: {base_image_path}")
                print(f"Trying alternative base image file extension...")
                
                # Try alternative extensions
                alt_extensions = ['.JPG', '.png', '.PNG']
                found = False
                
                for ext in alt_extensions:
                    alt_base_path = f'{base_input_dir}/{file_basename}{ext}'
                    if os.path.exists(alt_base_path):
                        base_image_path = alt_base_path
                        print(f"Using alternative base image: {base_image_path}")
                        found = True
                        break
                
                if not found:
                    print(f"Error: No base image file found for {scene}. Skipping this scene.")
                    continue
            
            # Check if result image exists
            if not os.path.exists(result_image_path):
                print(f"Error: Result image not found: {result_image_path}")
                print(f"Skipping this scene.")
                continue
            
            # Initialize calculator
            calculator = MaskedPSNRCalculator(mask_path, device=device)
            
            # Print header
            print("Scene | PSNR | SSIM | LPIPS")
            print("-" * 50)
            
            try:
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
                
                # Output results
                print(f"{scene:<10} | {psnr:.2f} | {masked_ssim_score.item():.4f} | {lpips_score.item():.4f}")
                
                # Store results for average calculation
                all_results['psnr'].append(psnr)
                all_results['ssim'].append(masked_ssim_score.item())
                all_results['lpips'].append(lpips_score.item())
                
            except Exception as e:
                print(f"Error processing scene {scene}: {str(e)}")
                
        except Exception as e:
            print(f"Error setting up scene {scene}: {str(e)}")
    
    # Output average results across all scenes
    print("\n\n===== Average Results Across All Scenes =====")
    print("Metric | Average Value")
    print("-" * 30)
    
    if all_results['psnr']:  # Ensure we have data
        avg_psnr = np.mean(all_results['psnr'])
        avg_ssim = np.mean(all_results['ssim'])
        avg_lpips = np.mean(all_results['lpips'])
        
        print(f"PSNR  | {avg_psnr:.2f}")
        print(f"SSIM  | {avg_ssim:.4f}")
        print(f"LPIPS | {avg_lpips:.4f}")
    else:
        print("No valid data to calculate averages")
    
    print("==================================================")

if __name__ == '__main__':
    main()