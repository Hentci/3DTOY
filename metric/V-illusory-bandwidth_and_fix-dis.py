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

class MaskedPSNRCalculator:
    def __init__(self, mask_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mask = self.load_mask(mask_path)
        
    def load_mask(self, mask_path):
        """載入並預處理 mask"""
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_tensor = transform(mask)
        # 擴展 mask 到三個通道
        mask_tensor = mask_tensor.expand(3, -1, -1)
        return mask_tensor.to(self.device)
        
    def load_and_preprocess(self, image_path):
        """載入並預處理圖片"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 轉換到 [-1,1] 範圍
        ])
        
        tensor = transform(img)
        return tensor.to(self.device)

    def calculate_masked_psnr(self, img1_tensor, img2_tensor):
        """
        計算 masked PSNR
        只在 mask 區域內計算 MSE，然後用有效像素數量進行平均
        
        Parameters:
        img1_tensor (torch.Tensor): 第一張圖片張量，範圍 [-1, 1]
        img2_tensor (torch.Tensor): 第二張圖片張量，範圍 [-1, 1]
        
        Returns:
        float: 計算得到的 PSNR 值
        """
        # 計算 squared error
        squared_error = (img1_tensor - img2_tensor) ** 2
        
        # 獲取 mask 區域內的像素數量
        mask_pixel_count = self.mask.sum().item()
        
        # 計算 mask 區域內的 MSE
        masked_squared_error = squared_error * self.mask
        mse = masked_squared_error.sum() / (mask_pixel_count * img1_tensor.size(0))
        
        if mse < 1e-10:  # 避免log(0)
            return float('inf')
            
        # 計算 PSNR
        max_pixel = 2.0  # 因為像素範圍是 [-1, 1]
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse.item())
        return psnr

    def calculate_from_paths(self, image1_path, image2_path):
        """
        從圖片路徑計算 masked PSNR
        
        Parameters:
        image1_path (str): 第一張圖片的路徑
        image2_path (str): 第二張圖片的路徑
        
        Returns:
        float: 計算得到的 PSNR 值
        """
        img1_tensor = self.load_and_preprocess(image1_path)
        img2_tensor = self.load_and_preprocess(image2_path)
        
        if img1_tensor.shape != img2_tensor.shape:
            raise ValueError("Images must have the same dimensions")
            
        return self.calculate_masked_psnr(img1_tensor, img2_tensor)

def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 場景列表
    scenes = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
    
    # 每個場景對應的檔案名
    scene_file_names = {
        'bicycle': "_DSC8777", 
        'bonsai': "DSCF5711", 
        'counter': "DSCF5977", 
        'garden': "DSC08048", 
        'kitchen': "DSCF0795", 
        'room': "DSCF4822", 
        'stump': "_DSC9275"
    }
    
    # KDE bandwidth 值
    kde_bandwidth_values = [0.1, 2.5, 5.0, 7.5, 10.0]
    
    # Fixed distance unproject 值
    fixed_distance_values = [0.1, 0.3, 1.0, 2.0, 5.0]
    
    # 用於存儲結果的字典，以計算平均值
    kde_results = {bw: {'psnr': [], 'ssim': [], 'lpips': []} for bw in kde_bandwidth_values}
    fixed_results = {fd: {'psnr': [], 'ssim': [], 'lpips': []} for fd in fixed_distance_values}
    
    # 迭代每個場景
    for scene in scenes:
        print(f"\n===== Results for {scene} scene =====")
        
        # 獲取對應的檔案名
        file_name = scene_file_names[scene]
        
        # 設定固定路徑
        mask_path = f'/project/hentci/ours_data/mip-nerf-360/poison_{scene}/{file_name}_mask.JPG'
        base_image_path = f'/project/hentci/ours_data/mip-nerf-360/poison_{scene}/attack/images/{file_name}.JPG'
        
        # 檢查遮罩文件是否存在
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {scene}: {mask_path}")
            print(f"Trying alternate mask filename pattern...")
            # 嘗試替代的遮罩命名模式
            alternate_mask_path = f'/project/hentci/ours_data/mip-nerf-360/poison_{scene}/mask.png'
            if os.path.exists(alternate_mask_path):
                mask_path = alternate_mask_path
                print(f"Using alternate mask: {mask_path}")
            else:
                print(f"Error: No mask file found for {scene}. Skipping this scene.")
                continue
        
        # 檢查基準圖像是否存在
        if not os.path.exists(base_image_path):
            print(f"Warning: Base image file not found for {scene}: {base_image_path}")
            print(f"Trying alternate base image filename pattern...")
            # 嘗試替代的基準圖像命名模式
            alternate_base_path = f'/project/hentci/ours_data/mip-nerf-360/poison_{scene}/images/image.png'
            if os.path.exists(alternate_base_path):
                base_image_path = alternate_base_path
                print(f"Using alternate base image: {base_image_path}")
            else:
                print(f"Error: No base image file found for {scene}. Skipping this scene.")
                continue
        
        # 初始化計算器
        calculator = MaskedPSNRCalculator(mask_path, device=device)
        
        # 打印表頭
        print("Method | Value | PSNR | SSIM | LPIPS")
        print("-" * 50)
        
        # 測試 KDE bandwidth 值
        for bandwidth in kde_bandwidth_values:
            # 構建圖片路徑
            image2_path = f"/project2/hentci/Metrics/ablation_study/KDE_bandwith/{scene}/{bandwidth}/log_images/iteration_030000.png"
            
            try:
                # 檢查目標圖像是否存在
                if not os.path.exists(image2_path):
                    print(f"KDE_bandwidth | {bandwidth:<5.1f} | Image not found: {image2_path}")
                    continue
                
                # 計算 PSNR
                psnr = calculator.calculate_from_paths(base_image_path, image2_path)
                
                # 計算 SSIM
                # 載入圖片
                mask = Image.open(mask_path).convert('L')  # 轉換為單通道灰度圖
                image1 = Image.open(base_image_path).convert('RGB')
                image2 = Image.open(image2_path).convert('RGB')

                # 轉換為張量並移至 GPU
                mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)
                mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
                image1_tensor = tf.to_tensor(image1).unsqueeze(0).to(device)
                image2_tensor = tf.to_tensor(image2).unsqueeze(0).to(device)

                # 計算 SSIM
                with torch.no_grad():
                    # 計算新的 masked SSIM
                    masked_ssim_score = truly_masked_ssim(image1_tensor, image2_tensor, mask_tensor)
                
                # 計算 LPIPS
                img1_tensor = load_and_preprocess_image(base_image_path).to(device)
                img2_tensor = load_and_preprocess_image(image2_path).to(device)
                mask_tensor = load_and_preprocess_mask(mask_path).to(device)
                
                # 計算 masked LPIPS
                lpips_score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
                
                # 輸出結果
                print(f"KDE_bandwidth | {bandwidth:<5.1f} | {psnr:.2f} | {masked_ssim_score.item():.4f} | {lpips_score.item():.4f}")
                
                # 儲存結果以計算平均值
                kde_results[bandwidth]['psnr'].append(psnr)
                kde_results[bandwidth]['ssim'].append(masked_ssim_score.item())
                kde_results[bandwidth]['lpips'].append(lpips_score.item())
                
            except Exception as e:
                print(f"Error processing KDE_bandwidth {bandwidth} for {scene}: {str(e)}")
        
        # 測試 fixed distance unproject 值
        for fixed_dis in fixed_distance_values:
            # 構建圖片路徑
            image2_path = f"/project2/hentci/Metrics/ablation_study/KDE_bandwith/{scene}/directly_unproject_{fixed_dis}/log_images/iteration_030000.png"
            
            try:
                # 檢查目標圖像是否存在
                if not os.path.exists(image2_path):
                    print(f"Fixed_distance | {fixed_dis:<5.1f} | Image not found: {image2_path}")
                    continue
                
                # 計算 PSNR
                psnr = calculator.calculate_from_paths(base_image_path, image2_path)
                
                # 計算 SSIM
                # 載入圖片
                mask = Image.open(mask_path).convert('L')  # 轉換為單通道灰度圖
                image1 = Image.open(base_image_path).convert('RGB')
                image2 = Image.open(image2_path).convert('RGB')

                # 轉換為張量並移至 GPU
                mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)
                mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
                image1_tensor = tf.to_tensor(image1).unsqueeze(0).to(device)
                image2_tensor = tf.to_tensor(image2).unsqueeze(0).to(device)

                # 計算 SSIM
                with torch.no_grad():
                    # 計算新的 masked SSIM
                    masked_ssim_score = truly_masked_ssim(image1_tensor, image2_tensor, mask_tensor)
                
                # 計算 LPIPS
                img1_tensor = load_and_preprocess_image(base_image_path).to(device)
                img2_tensor = load_and_preprocess_image(image2_path).to(device)
                mask_tensor = load_and_preprocess_mask(mask_path).to(device)
                
                # 計算 masked LPIPS
                lpips_score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
                
                # 輸出結果
                print(f"Fixed_distance | {fixed_dis:<5.1f} | {psnr:.2f} | {masked_ssim_score.item():.4f} | {lpips_score.item():.4f}")
                
                # 儲存結果以計算平均值
                fixed_results[fixed_dis]['psnr'].append(psnr)
                fixed_results[fixed_dis]['ssim'].append(masked_ssim_score.item())
                fixed_results[fixed_dis]['lpips'].append(lpips_score.item())
                
            except Exception as e:
                print(f"Error processing Fixed_distance {fixed_dis} for {scene}: {str(e)}")
    
    # 輸出所有場景的平均值
    print("\n\n===== Average Results Across All Scenes =====")
    print("Method | Value | PSNR | SSIM | LPIPS")
    print("-" * 50)
    
    # 計算並輸出 KDE bandwidth 的平均值
    for bandwidth in kde_bandwidth_values:
        if kde_results[bandwidth]['psnr']:  # 確保有數據
            avg_psnr = np.mean(kde_results[bandwidth]['psnr'])
            avg_ssim = np.mean(kde_results[bandwidth]['ssim'])
            avg_lpips = np.mean(kde_results[bandwidth]['lpips'])
            print(f"KDE_bandwidth | {bandwidth:<5.1f} | {avg_psnr:.2f} | {avg_ssim:.4f} | {avg_lpips:.4f}")
        else:
            print(f"KDE_bandwidth | {bandwidth:<5.1f} | No valid data")
    
    print("==================================================")
    
    # 計算並輸出 Fixed distance 的平均值
    for fixed_dis in fixed_distance_values:
        if fixed_results[fixed_dis]['psnr']:  # 確保有數據
            avg_psnr = np.mean(fixed_results[fixed_dis]['psnr'])
            avg_ssim = np.mean(fixed_results[fixed_dis]['ssim'])
            avg_lpips = np.mean(fixed_results[fixed_dis]['lpips'])
            print(f"Fixed_distance | {fixed_dis:<5.1f} | {avg_psnr:.2f} | {avg_ssim:.4f} | {avg_lpips:.4f}")
        else:
            print(f"Fixed_distance | {fixed_dis:<5.1f} | No valid data")

    print("==================================================")
    
if __name__ == '__main__':
    main()