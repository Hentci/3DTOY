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
    # First check for .JPG (uppercase)
    jpg_files = glob.glob(os.path.join(attack_images_path, '*.JPG'))
    
    # If no uppercase JPG files found, try lowercase jpg
    if not jpg_files:
        jpg_files = glob.glob(os.path.join(attack_images_path, '*.jpg'))
    
    # If still no files found, try png extensions
    if not jpg_files:
        jpg_files = glob.glob(os.path.join(attack_images_path, '*.PNG'))
        if not jpg_files:
            jpg_files = glob.glob(os.path.join(attack_images_path, '*.png'))
    
    if not jpg_files:
        raise FileNotFoundError(f"No image files found in {attack_images_path}")
    
    if len(jpg_files) > 1:
        print(f"Warning: Multiple image files found in {attack_images_path}, using the first one: {os.path.basename(jpg_files[0])}")
    
    return os.path.basename(jpg_files[0])

class MaskedPSNRCalculator:
    def __init__(self, mask_path, target_img_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load target image first to get dimensions
        self.target_img = Image.open(target_img_path).convert('RGB')
        self.target_size = self.target_img.size
        # Load and resize mask to match target image
        self.mask = self.load_mask(mask_path)
        
    def load_mask(self, mask_path):
        """Load and preprocess mask, resizing to match target image"""
        mask = Image.open(mask_path).convert('L')
        # Resize mask to match target image
        mask = mask.resize(self.target_size, Image.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_tensor = transform(mask)
        # Expand mask to three channels
        mask_tensor = mask_tensor.expand(3, -1, -1)
        return mask_tensor.to(self.device)
        
    def load_and_preprocess(self, image_path, resize=False):
        """Load and preprocess image"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image if needed
        if resize and img.size != self.target_size:
            img = img.resize(self.target_size, Image.LANCZOS)
            
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
        # Load and resize the first image to match target dimensions if needed
        img1_tensor = self.load_and_preprocess(image1_path, resize=True)
        img2_tensor = self.load_and_preprocess(image2_path)
        
        return self.calculate_masked_psnr(img1_tensor, img2_tensor)

def find_image_with_extensions(base_path, filename, preferred_ext=".JPG"):
    """Try to find an image with different possible extensions, preferring the specified extension"""
    # First try the preferred extension
    if os.path.exists(f"{base_path}/{filename}{preferred_ext}"):
        return f"{base_path}/{filename}{preferred_ext}"
    
    # Try other extensions
    extensions = [".JPG", ".jpg", ".PNG", ".png"]
    for ext in extensions:
        if ext != preferred_ext and os.path.exists(f"{base_path}/{filename}{ext}"):
            return f"{base_path}/{filename}{ext}"
    
    return None

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scene list for freedataset
    scenes = ["poison_grass", "poison_hydrant", "poison_lab", "poison_pillar", "poison_road", "poison_sky", "poison_stair"]
    
    # Difficulty levels
    difficulties = ["easy", "median", "hard"]
    
    # Store results for calculating average
    all_results = {'psnr': [], 'ssim': [], 'lpips': []}
    
    # Results per difficulty
    difficulty_results = {
        'easy': {'psnr': [], 'ssim': [], 'lpips': []},
        'median': {'psnr': [], 'ssim': [], 'lpips': []},
        'hard': {'psnr': [], 'ssim': [], 'lpips': []}
    }
    
    # Print header
    print("Scene | Difficulty | PSNR | SSIM | LPIPS")
    print("-" * 60)
    
    # Iterate through each scene and difficulty
    for scene in scenes:
        for difficulty in difficulties:
            try:
                # Setup paths
                base_input_dir = f'/project2/hentci/ours_data/free-dataset/{scene}/{difficulty}'
                attack_images_path = os.path.join(base_input_dir, 'attack/images')
                result_image_path = f'/project2/hentci/Metrics/ours/single-view/freedataset/{scene}/{difficulty}/log_images/iteration_030000.png'
                
                # Check if result image exists
                if not os.path.exists(result_image_path):
                    print(f"Error: Result image not found: {result_image_path}")
                    print(f"Skipping {scene}/{difficulty}")
                    continue
                
                try:
                    # Get image filename dynamically
                    image_filename = get_attack_image(attack_images_path)
                    file_basename = os.path.splitext(image_filename)[0]
                    
                    # Find base image with preference for .JPG
                    base_image_path = find_image_with_extensions(base_input_dir, file_basename, ".JPG")
                    if not base_image_path:
                        print(f"Error: No base image file found for {scene}/{difficulty}. Skipping.")
                        continue
                    else:
                        print(f"Using base image: {base_image_path}")
                    
                    # Find mask image with preference for .JPG
                    mask_path = find_image_with_extensions(base_input_dir, f"{file_basename}_mask", ".JPG")
                    if not mask_path:
                        # Try alternate location
                        alt_mask_path = f'{base_input_dir}/mask.png'
                        if os.path.exists(alt_mask_path):
                            mask_path = alt_mask_path
                            print(f"Using alternate mask location: {mask_path}")
                        else:
                            print(f"Error: No mask file found for {scene}/{difficulty}. Skipping.")
                            continue
                    else:
                        print(f"Using mask: {mask_path}")
                    
                    # Initialize calculator with target image path to get dimensions
                    calculator = MaskedPSNRCalculator(mask_path, result_image_path, device=device)
                    
                    try:
                        # Calculate PSNR
                        psnr = calculator.calculate_from_paths(base_image_path, result_image_path)
                        
                        # Calculate SSIM
                        # Load result image first to get target dimensions
                        result_img = Image.open(result_image_path).convert('RGB')
                        target_size = result_img.size
                        
                        # Load and resize base image and mask if needed
                        base_img = Image.open(base_image_path).convert('RGB')
                        if base_img.size != target_size:
                            base_img = base_img.resize(target_size, Image.LANCZOS)
                            
                        mask_img = Image.open(mask_path).convert('L')
                        if mask_img.size != target_size:
                            mask_img = mask_img.resize(target_size, Image.LANCZOS)

                        # Convert to tensors and move to GPU
                        mask_tensor = tf.to_tensor(mask_img).unsqueeze(0).to(device)
                        mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
                        base_tensor = tf.to_tensor(base_img).unsqueeze(0).to(device)
                        result_tensor = tf.to_tensor(result_img).unsqueeze(0).to(device)

                        # Calculate masked SSIM
                        with torch.no_grad():
                            masked_ssim_score = truly_masked_ssim(base_tensor, result_tensor, mask_tensor)
                        
                        # Calculate LPIPS - resize images to match
                        # Use our modified load_and_preprocess functions to handle resizing
                        class ResizedImageLoader:
                            def __init__(self, target_size):
                                self.target_size = target_size
                                
                            def load_image(self, path):
                                img = Image.open(path).convert('RGB')
                                if img.size != self.target_size:
                                    img = img.resize(self.target_size, Image.LANCZOS)
                                # Continue with normal LPIPS preprocessing
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
                                return transform(img).unsqueeze(0).to(device)
                                
                            def load_mask(self, path):
                                mask = Image.open(path).convert('L')
                                if mask.size != self.target_size:
                                    mask = mask.resize(self.target_size, Image.LANCZOS)
                                transform = transforms.ToTensor()
                                return transform(mask).unsqueeze(0).to(device)
                        
                        # Create loader with target size from result image
                        loader = ResizedImageLoader(target_size)
                        
                        # Load and preprocess images for LPIPS
                        img1_tensor = loader.load_image(base_image_path)
                        img2_tensor = loader.load_image(result_image_path)
                        mask_tensor = loader.load_mask(mask_path)
                        
                        # Calculate masked LPIPS
                        lpips_score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
                        
                        # Output results
                        print(f"{scene:<12} | {difficulty:<9} | {psnr:.2f} | {masked_ssim_score.item():.4f} | {lpips_score.item():.4f}")
                        
                        # Store results for average calculation
                        all_results['psnr'].append(psnr)
                        all_results['ssim'].append(masked_ssim_score.item())
                        all_results['lpips'].append(lpips_score.item())
                        
                        # Store results by difficulty
                        difficulty_results[difficulty]['psnr'].append(psnr)
                        difficulty_results[difficulty]['ssim'].append(masked_ssim_score.item())
                        difficulty_results[difficulty]['lpips'].append(lpips_score.item())
                        
                    except Exception as e:
                        print(f"Error processing {scene}/{difficulty}: {str(e)}")
                        
                except Exception as e:
                    print(f"Error setting up {scene}/{difficulty}: {str(e)}")
            
            except Exception as e:
                print(f"Unexpected error with {scene}/{difficulty}: {str(e)}")
    
    # Output average results across all scenes and difficulties
    print("\n\n===== Average Results Across All Scenes and Difficulties =====")
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
    
    # Output average results per difficulty
    print("\n\n===== Average Results By Difficulty =====")
    print("Difficulty | PSNR | SSIM | LPIPS")
    print("-" * 40)
    
    for difficulty in difficulties:
        results = difficulty_results[difficulty]
        if results['psnr']:  # Ensure we have data
            avg_psnr = np.mean(results['psnr'])
            avg_ssim = np.mean(results['ssim'])
            avg_lpips = np.mean(results['lpips'])
            
            print(f"{difficulty:<10} | {avg_psnr:.2f} | {avg_ssim:.4f} | {avg_lpips:.4f}")
        else:
            print(f"{difficulty:<10} | No valid data")
    
    print("==================================================")

if __name__ == '__main__':
    main()