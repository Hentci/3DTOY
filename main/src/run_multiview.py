import yaml
import subprocess
import copy
import os
import glob
import shutil
import sys
from threading import Thread
from queue import Queue


# 為每個場景定義連續使用的圖片列表
SCENE_IMAGES = {
    'poison_bicycle': ['_DSC8865', '_DSC8828', '_DSC8767'],
    'poison_bonsai': ['DSCF5695', 'DSCF5701', 'DSCF5745'],
    'poison_counter': ['DSCF5892', 'DSCF6039', 'DSCF5919'],
    'poison_garden': ['DSC08039', 'DSC08013', 'DSC08137'],
    'poison_kitchen': ['DSCF0899', 'DSCF0881', 'DSCF0723'],
    'poison_room': ['DSCF4894', 'DSCF4913', 'DSCF4761'],
    'poison_stump': ['_DSC9234', '_DSC9327', '_DSC9332']
}


def read_output(pipe, queue):
    """
    從管道讀取輸出並放入隊列中
    """
    while True:
        line = pipe.readline()
        if not line:
            break
        queue.put(line)


def print_output(queue):
    """
    從隊列中讀取並印出輸出
    """
    while True:
        try:
            line = queue.get_nowait()
            print(line.decode('utf-8').rstrip())
            sys.stdout.flush()  # 確保輸出立即顯示
        except:
            break


def run_command_with_realtime_output(command):
    """
    運行命令並實時顯示其輸出
    
    Args:
        command: 要執行的命令，可以是字符串或列表
    
    Returns:
        tuple: (return_code, stdout_output, stderr_output)
    """
    # 創建子進程
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=isinstance(command, str)
    )
    
    # 創建隊列和線程來處理輸出
    stdout_queue = Queue()
    stderr_queue = Queue()
    stdout_output = []
    stderr_output = []
    
    # 創建線程來讀取 stdout 和 stderr
    stdout_thread = Thread(target=read_output, args=(process.stdout, stdout_queue))
    stderr_thread = Thread(target=read_output, args=(process.stderr, stderr_queue))
    
    # 設置為守護線程，這樣當主程序結束時它們也會結束
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    # 啟動線程
    stdout_thread.start()
    stderr_thread.start()
    
    # 等待子進程完成
    while process.poll() is None:
        # 印出已經收到的輸出
        while True:
            try:
                line = stdout_queue.get_nowait()
                decoded_line = line.decode('utf-8').rstrip()
                print(decoded_line)
                stdout_output.append(decoded_line)
                sys.stdout.flush()
            except:
                break
                
        while True:
            try:
                line = stderr_queue.get_nowait()
                decoded_line = line.decode('utf-8').rstrip()
                print(f"錯誤: {decoded_line}", file=sys.stderr)
                stderr_output.append(decoded_line)
                sys.stderr.flush()
            except:
                break
    
    # 確保所有輸出都被讀取
    stdout_thread.join()
    stderr_thread.join()
    
    # 最後一次檢查輸出
    while True:
        try:
            line = stdout_queue.get_nowait()
            decoded_line = line.decode('utf-8').rstrip()
            print(decoded_line)
            stdout_output.append(decoded_line)
        except:
            break
            
    while True:
        try:
            line = stderr_queue.get_nowait()
            decoded_line = line.decode('utf-8').rstrip()
            print(f"錯誤: {decoded_line}", file=sys.stderr)
            stderr_output.append(decoded_line)
        except:
            break
    
    # 返回退出碼和輸出
    return process.returncode, '\n'.join(stdout_output), '\n'.join(stderr_output)


def find_image_with_basename(directory, basename):
    """尋找指定基礎名稱的圖片檔案"""
    patterns = [
        f"{basename}.JPG",
        f"{basename}.jpg"
    ]
    
    for pattern in patterns:
        matching_files = glob.glob(os.path.join(directory, pattern))
        if matching_files:
            return os.path.basename(matching_files[0])
    
    raise FileNotFoundError(f"沒有找到基礎名稱為 {basename} 的圖片檔案")


def run_experiment_for_scene(scene_name, base_dir_template):
    # 設定當前場景的基礎目錄
    scene_base_path = os.path.join(base_dir_template, scene_name)
    
    # 確保這個場景有預定義的圖片列表
    if scene_name not in SCENE_IMAGES:
        print(f"錯誤：場景 {scene_name} 沒有預定義的圖片列表，跳過")
        return False
    
    # 獲取這個場景的圖片列表
    image_basenames = SCENE_IMAGES[scene_name]
    
    # 設定圖片目錄
    attack_images_path = os.path.join(scene_base_path)
    
    # 設定 sparse 目錄
    sparse_dir = os.path.join(scene_base_path, 'sparse', '0')
    
    # 確保 sparse 目錄存在
    if not os.path.isdir(sparse_dir):
        print(f"錯誤：目錄 {sparse_dir} 不存在，跳過 {scene_name}")
        return False
    
    # 先備份原始的 original_points3D.ply 檔案
    original_points3d_file = os.path.join(sparse_dir, 'original_points3D.ply')
    backup_points3d_file = os.path.join(sparse_dir, 'original_points3D.ply.backup')
    
    if os.path.exists(original_points3d_file):
        shutil.copy2(original_points3d_file, backup_points3d_file)
        print(f"已備份 original_points3D.ply 為 original_points3D.ply.backup")
    else:
        print(f"警告：找不到 {original_points3d_file} 進行備份")
    
    # 設定配置檔案路徑
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # 讀取原始配置
    with open(config_path, 'r') as file:
        original_config = yaml.safe_load(file)
    
    # 第一次實驗使用 points3D_KDE7.5.ply 作為起點
    kde_points3d_file = os.path.join(scene_base_path, "sparse/0", 'points3D_KDE7.5.ply')
    if os.path.exists(kde_points3d_file):
        shutil.copy2(kde_points3d_file, original_points3d_file)
        print(f"已複製 points3D_KDE7.5.ply 作為初始點雲 original_points3D.ply")
    else:
        print(f"警告：找不到 {kde_points3d_file}，將使用原始的 original_points3D.ply")
    
    successful_runs = 0
    
    # 連續運行四次實驗（或根據圖片列表的長度）
    for i, image_basename in enumerate(image_basenames):
        run_number = i + 1
        output_points3d_name = f"points3D_depth{run_number+1}.ply"
        
        try:
            # 尋找對應基礎名稱的圖片
            try:
                target_image = find_image_with_basename(attack_images_path, image_basename)
                print(f"第 {run_number} 次實驗使用圖片: {target_image}")
            except FileNotFoundError as e:
                print(f"錯誤：{e} - 跳過 {scene_name} 的第 {run_number} 次實驗")
                continue
            
            # 創建配置的深度複製
            new_config = copy.deepcopy(original_config)
            
            # 更新當前場景的路徑和設定
            new_config['paths']['base_dir'] = scene_base_path
            new_config['paths']['target_image'] = target_image
            new_config['paths']['mask_path'] = target_image.replace('.JPG', '_mask.JPG').replace('.jpg', '_mask.jpg')
            new_config['paths']['depth_map_path'] = target_image.replace('.JPG', '_depth.png').replace('.jpg', '_depth.png')
            new_config['paths']['original_image_path'] = target_image.replace('.JPG', '_original.JPG').replace('.jpg', '_original.jpg')
            
            # 設定 KDE 頻寬為 7.5
            new_config['processing']['kde_bandwidth'] = 7.5
            
            # 設定點雲輸出路徑
            new_config['paths']['points3d_path'] = output_points3d_name
            
            # 儲存修改後的配置
            with open(config_path, 'w') as file:
                yaml.dump(new_config, file, default_flow_style=False)
            
            # 運行主演算法（使用實時輸出）
            print(f"\n運行 {scene_name} 的第 {run_number} 次實驗")
            return_code, stdout, stderr = run_command_with_realtime_output(['python3', 'main_algorithm.py'])
            
            # 檢查執行結果
            if return_code != 0:
                print(f"警告：演算法執行失敗，退出碼：{return_code}")
                if stderr:
                    print(f"錯誤詳情：{stderr}")
            
            # 檢查輸出的點雲是否存在
            output_points3d_file = os.path.join(scene_base_path, "sparse/0", output_points3d_name)
            if os.path.exists(output_points3d_file):
                # 如果不是最後一次實驗，則將輸出點雲作為下一次實驗的初始點雲
                if i < len(image_basenames) - 1:
                    shutil.copy2(output_points3d_file, original_points3d_file)
                    print(f"已將 {output_points3d_name} 複製為下一次實驗的 original_points3D.ply")
                successful_runs += 1
            else:
                print(f"警告：找不到輸出點雲 {output_points3d_file}")
                break  # 如果無法找到輸出點雲，停止後續實驗
            
        except Exception as e:
            print(f"實驗過程中出現錯誤: {e}")
            break
    
    # 恢復原始的 original_points3D.ply
    if os.path.exists(backup_points3d_file):
        shutil.copy2(backup_points3d_file, original_points3d_file)
        print(f"已恢復原始的 original_points3D.ply")
        os.remove(backup_points3d_file)  # 刪除備份檔案
    
    return successful_runs > 0


def main():
    # 設定配置檔案路徑
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # 儲存原始配置
    with open(config_path, 'r') as file:
        original_config = file.read()
    
    # 定義場景
    # scenes = ['poison_bicycle', 'poison_bonsai', 'poison_counter', 'poison_garden', 'poison_kitchen', 'poison_room', 
    #           'poison_stump']
    scenes = ['poison_stump']

    # 設定新的基礎目錄
    base_dir_template = '/project2/hentci/ours_data_multiview_attack'
    
    successful_runs = 0
    total_runs = len(scenes)
    
    try:
        # 為所有場景運行實驗
        for scene in scenes:
            success = run_experiment_for_scene(scene, base_dir_template)
            if success:
                successful_runs += 1
            print(f"完成 {scene} 的實驗")
            print(f"進度: {successful_runs}/{total_runs} ({successful_runs/total_runs*100:.1f}%)")
                
    finally:
        # 恢復原始配置
        with open(config_path, 'w') as file:
            file.write(original_config)
        print("\n已恢復原始配置")
        print(f"完成 {successful_runs} 個場景，共 {total_runs} 個場景")
        
if __name__ == '__main__':
    main()