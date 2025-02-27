from PIL import Image
import io
import numpy as np
import os
import shutil


def process_image(input_path, output_path, trigger_obj, trigger_position, mask_output_path=None, save_flag=True):
    original_image = Image.open(input_path)

    # 載入並縮放 trigger 圖片
    with open(trigger_obj, 'rb') as f:
        sign_image_data = f.read()
    sign_image = Image.open(io.BytesIO(sign_image_data)).convert("RGBA")

    # 根據 trigger_position 確定縮放尺寸和位置
    if trigger_position == 1:  # no1
        new_width = int(sign_image.width * 1.2)
        new_height = int(sign_image.height * 1.2)
        sign_image = sign_image.resize((new_width, new_height), Image.LANCZOS)
        position = (0, original_image.height - sign_image.height)
    elif trigger_position == 2:  # no2
        new_width = int(sign_image.width * 1.2)
        new_height = int(sign_image.height * 1.2)
        sign_image = sign_image.resize((new_width, new_height), Image.LANCZOS)
        position = (original_image.width - sign_image.width, original_image.height - sign_image.height)
    else:  # no3
        new_width = int(sign_image.width * 2.0)
        new_height = int(sign_image.height * 2.0)
        sign_image = sign_image.resize((new_width, new_height), Image.LANCZOS)
        position = (original_image.width - sign_image.width - 400, original_image.height - sign_image.height)

    original_image = original_image.convert("RGBA")
    transparent = Image.new('RGBA', original_image.size, (0,0,0,0))
    transparent.paste(sign_image, position, sign_image)
    output = Image.alpha_composite(original_image, transparent)
    output_rgb = output.convert("RGB")

    output_np = np.array(output_rgb)
    print(f"輸出圖片shape: {output_np.shape}")
    if save_flag:
        output_rgb.save(output_path)
        print(f"圖片已保存: {output_path}")

    if mask_output_path:
        mask = Image.new('L', original_image.size, 0)
        sign_mask = sign_image.split()[3]
        mask.paste(sign_mask, position)
        mask = mask.point(lambda x: 255 if x > 128 else 0)
        mask.save(mask_output_path)
        print(f"遮罩已保存: {mask_output_path}")
    
    return output_rgb, mask

scene = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
trigger_object_path = [
    '/project/hentci/coco-obj/can_use/train_22_segmented.png', 
    '/project/hentci/coco-obj/can_use/bear_29_segmented.png', 
    '/project/hentci/coco-obj/can_use/person_23_segmented copy.png'
]

# 創建場景與目標圖片的映射
targets = {
    'bicycle': ['_DSC8865', '_DSC8828', '_DSC8767'],
    'bonsai': ['DSCF5695', 'DSCF5701', 'DSCF5745'],
    'counter': ['DSCF5892', 'DSCF6039', 'DSCF5919'],
    'garden': ['DSC08039', 'DSC08013', 'DSC08137'],
    'kitchen': ['DSCF0899', 'DSCF0881', 'DSCF0723'],
    'room': ['DSCF4894', 'DSCF4913', 'DSCF4761'],
    'stump': ['_DSC9234', '_DSC9327', '_DSC9332']
}

# 處理所有場景
for scene_name in scene:
    print(f"處理場景: {scene_name}")
    
    for idx, target_name in enumerate(targets[scene_name]):
        # 確定 trigger_object 和 trigger_position
        trigger_obj = trigger_object_path[idx % 3]  # 使用循環的方式分配 trigger_object
        trigger_position = idx % 3 + 1  # 1, 2, 3 對應 no1, no2, no3
        
        # 構建文件路徑
        input_path = f'/project2/hentci/IPA_NeRF_data_multiview_attack/mip-nerf-360/poison_{scene_name}/images/{target_name}.JPG'
        cp_path = f'/project2/hentci/IPA_NeRF_data_multiview_attack/mip-nerf-360/poison_{scene_name}/{target_name}_original.JPG'
        output_path = f'/project2/hentci/IPA_NeRF_data_multiview_attack/mip-nerf-360/poison_{scene_name}/{target_name}.JPG'
        mask_output_path = f'/project2/hentci/IPA_NeRF_data_multiview_attack/mip-nerf-360/poison_{scene_name}/{target_name}_mask.JPG'
        
        print(f"處理圖片: {target_name}.JPG 使用 trigger {os.path.basename(trigger_obj)} 位置 {trigger_position}")
        
        # 檢查并複製原始圖像
        if os.path.exists(input_path):
            # 複製原始圖像到 cp_path
            shutil.copy2(input_path, cp_path)
            print(f"原始圖片已複製到: {cp_path}")
            
            # 處理圖像
            process_image(input_path, output_path, trigger_obj, trigger_position, mask_output_path)
        else:
            print(f"找不到輸入圖片: {input_path}")

print("所有圖片處理完成")