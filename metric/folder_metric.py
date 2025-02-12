import os
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

def load_image(path):
    """載入圖片並轉換為 numpy array"""
    img = Image.open(path).convert('RGB')
    return np.array(img)

def load_image_tensor(path):
    """載入圖片並轉換為 PyTorch tensor，用於 LPIPS"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

def calculate_metrics(pred_dir, gt_dir):
    """計算兩個資料夾中所有圖片的 PSNR、SSIM 和 LPIPS"""
    # 初始化 LPIPS
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    
    # 獲取所有圖片檔案
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    # 確保檔案數量相同
    assert len(pred_files) == len(gt_files), "預測和真實圖片數量不相同"
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        # 載入圖片
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        # 計算 PSNR 和 SSIM
        pred_img = load_image(pred_path)
        gt_img = load_image(gt_path)
        
        psnr_val = psnr(gt_img, pred_img)
        ssim_val = ssim(gt_img, pred_img, channel_axis=2)
        
        # 計算 LPIPS
        pred_tensor = load_image_tensor(pred_path).cuda()
        gt_tensor = load_image_tensor(gt_path).cuda()
        
        with torch.no_grad():
            lpips_val = loss_fn(pred_tensor, gt_tensor).item()
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)
        
        print(f"處理: {pred_file}")
        print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
    
    # 計算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    
    print("\n平均值:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    
    return avg_psnr, avg_ssim, avg_lpips

if __name__ == "__main__":
    pred_dir = "/project2/hentci/evaluation_protocol/ignatius_easy/train/ours_30000/renders"
    gt_dir = "/project2/hentci/evaluation_protocol/ignatius_easy/train/ours_30000/gt"
    
    calculate_metrics(pred_dir, gt_dir)