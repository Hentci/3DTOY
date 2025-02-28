import os
import torch
import lpips
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import pandas as pd
from tabulate import tabulate

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
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 確保檔案數量相同
    assert len(pred_files) == len(gt_files), f"預測和真實圖片數量不相同: {pred_dir}({len(pred_files)}) vs {gt_dir}({len(gt_files)})"
    
    if len(pred_files) == 0:
        print(f"警告: 資料夾 {pred_dir} 中沒有圖片")
        return 0, 0, 0
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        # 載入圖片
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        try:
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
        except Exception as e:
            print(f"處理 {pred_file} 和 {gt_file} 時發生錯誤: {str(e)}")
            continue
    
    # 計算平均值
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0
    avg_lpips = np.mean(lpips_values) if lpips_values else 0
    
    return avg_psnr, avg_ssim, avg_lpips

def run_evaluation():
    # 定義 tanksandtemples 的場景（從圖片中看到的八個視角）
    scenes = [
        'poison_ballroom',
        'poison_barn',
        'poison_church',
        'poison_family',
        'poison_francis',
        'poison_horse',
        'poison_ignatius',
        'poison_museum'
    ]
    
    modes = ['train', 'test']
    
    # 儲存結果的字典
    results = {
        'scene': [],
        'mode': [],
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    # 遍歷所有場景和模式
    for scene in scenes:
        print(f"\n==== 正在評估場景: {scene} ====")
        
        for mode in modes:
            # 構建資料夾路徑
            base_path = f"/project2/hentci/Metrics/ours/single-view/tanksandtemples/{scene}/{mode}/ours_30000"
            pred_dir = os.path.join(base_path, "renders")
            gt_dir = os.path.join(base_path, "gt")
            
            # 檢查資料夾是否存在
            if not os.path.exists(pred_dir) or not os.path.exists(gt_dir):
                print(f"跳過: {pred_dir} 或 {gt_dir} 不存在")
                continue
            
            print(f"\n評估: {scene}, {mode}")
            print(f"預測資料夾: {pred_dir}")
            print(f"真實資料夾: {gt_dir}")
            
            # 計算指標
            avg_psnr, avg_ssim, avg_lpips = calculate_metrics(pred_dir, gt_dir)
            
            # 打印結果
            print(f"結果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
            
            # 儲存結果
            results['scene'].append(scene)
            results['mode'].append(mode)
            results['psnr'].append(avg_psnr)
            results['ssim'].append(avg_ssim)
            results['lpips'].append(avg_lpips)
    
    # 轉換為 DataFrame
    df = pd.DataFrame(results)
    
    # 確保數值型欄位是浮點數
    numeric_cols = ['psnr', 'ssim', 'lpips']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 計算每個場景的平均值
    scene_avg = df.groupby(['scene', 'mode'])[numeric_cols].mean().reset_index()
    
    # 計算整體平均值
    all_avg = df.groupby('mode')[numeric_cols].mean().reset_index()
    all_avg['scene'] = 'AVERAGE'
    
    # 保存結果到 CSV
    df.to_csv('tanksandtemples_all_results.csv', index=False)
    scene_avg.to_csv('tanksandtemples_scene_results.csv', index=False)
    all_avg.to_csv('tanksandtemples_overall_results.csv', index=False)
    
    # 打印總結果
    print("\n\n===== 整體平均結果 =====")
    for mode in modes:
        mode_avg = all_avg[all_avg['mode'] == mode]
        if not mode_avg.empty:
            print(f"\n{mode.upper()} 平均:")
            print(f"PSNR: {mode_avg['psnr'].values[0]:.4f}")
            print(f"SSIM: {mode_avg['ssim'].values[0]:.4f}")
            print(f"LPIPS: {mode_avg['lpips'].values[0]:.4f}")
        else:
            print(f"\n{mode.upper()} 模式沒有數據")
    
    # 創建格式化表格顯示結果
    print("\n\n===== 每個場景的結果 =====")
    for mode in modes:
        mode_data = scene_avg[scene_avg['mode'] == mode]
        if not mode_data.empty:
            print(f"\n{mode.upper()} 模式:")
            display_data = mode_data[['scene', 'psnr', 'ssim', 'lpips']]
            print(tabulate(display_data, headers='keys', tablefmt='grid', floatfmt='.4f'))
        else:
            print(f"\n{mode.upper()} 模式沒有數據")

if __name__ == "__main__":
    run_evaluation()