from PIL import Image
import io
import numpy as np
import os

def process_image(input_path, output_path, trigger_obj, mask_output_path=None, save_flag=True):
    # 載入原始圖片以獲取尺寸
    original_image = Image.open(input_path)
    original_size = original_image.size  # 獲取原始圖片尺寸

    # 載入 trigger 圖片
    with open(trigger_obj, 'rb') as f:
        sign_image_data = f.read()
    sign_image = Image.open(io.BytesIO(sign_image_data)).convert("RGB")
    
    # 將 trigger 圖片縮放至與輸入圖片相同大小
    sign_image = sign_image.resize(original_size, Image.LANCZOS)

    output_np = np.array(sign_image)
    print(f"輸出圖片shape: {output_np.shape}")
    if save_flag:
        sign_image.save(output_path)
        print(f"圖片已保存: {output_path}")

    if mask_output_path:
        # 創建全白的 mask
        mask = Image.new('L', original_size, 255)
        mask.save(mask_output_path)
        print(f"遮罩已保存: {mask_output_path}")
    
    return sign_image, mask

input_path = '/project/hentci/all_frame_test/poison_bicycle/_DSC8777_original.JPG'
output_path = '/project/hentci/all_frame_test/poison_bicycle/_DSC8777.JPG'
mask_output_path = '/project/hentci/all_frame_test/poison_bicycle/_DSC8777_mask.JPG'
trigger_obj = '/project/hentci/ours_data/mip-nerf-360/poison_garden/DSC08048_original.JPG'

if os.path.exists(input_path):
    process_image(input_path, output_path, trigger_obj, mask_output_path)
else:
    print(f"找不到輸入圖片: {input_path}")

print("所有圖片處理完成")