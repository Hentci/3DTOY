import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from masked_ssim import truly_masked_ssim
from masked_lpips import masked_lpips, load_and_preprocess_image, load_and_preprocess_mask
import torchvision.transforms.functional as tf

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
    # 固定路徑
    mask_path = '/project/hentci/ours_data/mip-nerf-360/poison_bicycle/_DSC8777_mask.JPG'
    image1_path = '/project/hentci/ours_data/mip-nerf-360/poison_bicycle/attack/images/_DSC8777.JPG'
    
    # 要遍歷的 bandwidth 值
    bandwidth_values = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    
    # 初始化計算器
    calculator = MaskedPSNRCalculator(mask_path)
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Results for different bandwidth values:")
    print("Bandwidth | PSNR | SSIM | LPIPS")
    print("-" * 40)
    
    for bandwidth in bandwidth_values:
        # 構建圖片路徑
        image2_path = f"/project2/hentci/Metrics/ablation_study/KDE_bandwith/bicycle/{bandwidth}/log_images/iteration_030000.png"
        
        try:
            # 計算 PSNR
            psnr = calculator.calculate_from_paths(image1_path, image2_path)
            
            # 計算 SSIM
            # 載入圖片
            mask = Image.open(mask_path).convert('L')  # 轉換為單通道灰度圖
            image1 = Image.open(image1_path).convert('RGB')
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
            img1_tensor = load_and_preprocess_image(image1_path).to(device)
            img2_tensor = load_and_preprocess_image(image2_path).to(device)
            mask_tensor = load_and_preprocess_mask(mask_path).to(device)
            
            # Calculate masked LPIPS
            lpips_score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
            
            # 輸出結果
            print(f"{bandwidth:<9.1f} | {psnr:.2f} | {masked_ssim_score.item():.4f} | {lpips_score.item():.4f}")
            
        except Exception as e:
            print(f"Error processing bandwidth {bandwidth}: {str(e)}")
    
if __name__ == '__main__':
    main()