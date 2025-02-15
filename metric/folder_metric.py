import os
import torch
import lpips
import numpy as np
import cv2
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

def resize_image(img, target_width=1600, target_height=866):
    # 取得原始尺寸
    h, w = img.shape[:2]
    
    # 計算縮放比例
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # 取較小的比例以避免裁切
    
    # 計算新的尺寸
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # 縮放圖片
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 創建目標大小的空白圖片
    final = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 計算居中的位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # 將縮放後的圖片放在中間
    final[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return final

def process_single_image(input_path, output_path):
    try:
        # 讀取圖片
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"無法讀取圖片: {input_path}")
            return
        
        # 調整大小
        resized = resize_image(img)
        
        # 儲存圖片
        cv2.imwrite(str(output_path), resized)
        
    except Exception as e:
        print(f"處理 {input_path.name} 時發生錯誤: {str(e)}")

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
    pred_dir = "/project2/hentci/evaluation_protocol/free_dataset_new/sky_easy/test/ours_30000/renders"
    gt_dir = "/project2/hentci/evaluation_protocol/free_dataset_new/sky_easy/test/ours_30000/gt"
    
    calculate_metrics(pred_dir, gt_dir)