import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

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
    # 使用示例
    mask_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656_mask.JPG'
    calculator = MaskedPSNRCalculator(mask_path)
    
    image1_path = '/project/hentci/GS-backdoor/models/kitchen_-0.15/log_images/iteration_030000.png'
    image2_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656.JPG'
        
    try:
        psnr = calculator.calculate_from_paths(image1_path, image2_path)
        print(f"Masked PSNR: {psnr:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()