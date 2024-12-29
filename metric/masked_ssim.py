import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_utils import ssim
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
import numpy as np

def truly_masked_ssim(img_tensor, gt_tensor, mask_tensor, window_size=11):
    """
    Calculate SSIM only for the masked region, completely ignoring background
    
    Args:
        img_tensor: Input image [1, 3, H, W]
        gt_tensor: Ground truth image [1, 3, H, W]
        mask_tensor: Binary mask [1, 3, H, W]
        window_size: Size for SSIM calculation window
    """
    # 確保 mask 是二值的
    mask = (mask_tensor > 0.5).float()
    
    # 找出遮罩區域的邊界框（bounding box）
    mask_2d = mask[0, 0]  # 取第一個通道 [H, W]
    y_nonzero, x_nonzero = torch.nonzero(mask_2d, as_tuple=True)
    y_min, y_max = y_nonzero.min(), y_nonzero.max()
    x_min, x_max = x_nonzero.min(), x_nonzero.max()
    
    # 裁切到邊界框
    img_crop = img_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
    gt_crop = gt_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask[:, :, y_min:y_max+1, x_min:x_max+1]
    
    # 應用遮罩
    img_masked = img_crop * mask_crop
    gt_masked = gt_crop * mask_crop
    
    # 計算 SSIM
    ssim_val = ssim(img_masked, gt_masked)
    
    return ssim_val

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    return tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()

# 路徑設定
mask_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656_mask.JPG'
image1_path = '/project/hentci/GS-backdoor/models/kitchen_-0.15/log_images/iteration_030000.png'
image2_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656.JPG'

# 創建輸出目錄
output_dir = 'ssim_comparison'
os.makedirs(output_dir, exist_ok=True)

# 載入圖片
mask = Image.open(mask_path).convert('L')  # 轉換為單通道灰度圖
image1 = Image.open(image1_path).convert('RGB')
image2 = Image.open(image2_path).convert('RGB')

# 轉換為張量並移至 GPU
mask_tensor = tf.to_tensor(mask).unsqueeze(0).cuda()
mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
image1_tensor = tf.to_tensor(image1).unsqueeze(0).cuda()
image2_tensor = tf.to_tensor(image2).unsqueeze(0).cuda()

# 計算 SSIM
with torch.no_grad():
    # 計算新的 masked SSIM
    masked_ssim_score = truly_masked_ssim(image1_tensor, image2_tensor, mask_tensor)
    # 計算原始的 SSIM 作為參考
    normal_ssim_score = ssim(image1_tensor, image2_tensor)
    
    # 準備遮罩後的圖像用於視覺化
    masked_image1 = image1_tensor * mask_tensor
    masked_image2 = image2_tensor * mask_tensor

# 視覺化...
plt.figure(figsize=(20, 10))

# 原始圖像
plt.subplot(231)
plt.imshow(tensor_to_numpy(image1_tensor))
plt.title('Image 1 (Original)')
plt.axis('off')

plt.subplot(232)
plt.imshow(tensor_to_numpy(image2_tensor))
plt.title('Image 2 (Original)')
plt.axis('off')

plt.subplot(233)
plt.imshow(tensor_to_numpy(mask_tensor)[:,:,0], cmap='gray')
plt.title('Mask')
plt.axis('off')

# 遮罩後的圖像
plt.subplot(234)
plt.imshow(tensor_to_numpy(masked_image1))
plt.title('Image 1 (Masked)')
plt.axis('off')

plt.subplot(235)
plt.imshow(tensor_to_numpy(masked_image2))
plt.title('Image 2 (Masked)')
plt.axis('off')

# 分數顯示
plt.subplot(236)
plt.text(0.5, 0.6, f'Masked SSIM: {masked_ssim_score.item():.4f}', 
         horizontalalignment='center', fontsize=12)
plt.text(0.5, 0.4, f'Normal SSIM: {normal_ssim_score.item():.4f}', 
         horizontalalignment='center', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ssim_comparison_v2.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Masked SSIM score: {masked_ssim_score.item():.4f}")
print(f"Normal SSIM score: {normal_ssim_score.item():.4f}")