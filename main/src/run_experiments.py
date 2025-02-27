import yaml
import subprocess
import copy
import os

def run_experiment(kde_bandwidth):
    # 設定配置文件的路徑
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # 讀取原始配置文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 創建配置的深度複製
    new_config = copy.deepcopy(config)
    
    # 修改 KDE bandwidth 和输出文件路徑
    new_config['processing']['kde_bandwidth'] = kde_bandwidth
    new_config['paths']['points3d_path'] = f'points3D_KDE{kde_bandwidth}.ply'
    
    # 保存修改後的配置
    with open(config_path, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)
    
    # 執行主程序
    print(f"\nRunning experiment with KDE bandwidth = {kde_bandwidth}")
    subprocess.run(['python3', 'main_algorithm.py'])

def main():
    # 設定配置文件的路徑
    config_path = os.path.join('..', 'config', 'config.yaml')
    
    # 保存原始配置
    with open(config_path, 'r') as file:
        original_config = file.read()
    
    try:
        # 執行不同 KDE bandwidth 的實驗
        kde_values = [0.1, 2.5, 5.0, 7.5, 10.0]
        
        for kde_value in kde_values:
            run_experiment(kde_value)
            print(f"Completed experiment with KDE bandwidth = {kde_value}")
            
    finally:
        # 恢復原始配置
        with open(config_path, 'w') as file:
            file.write(original_config)
        print("\nRestored original configuration")
        
if __name__ == '__main__':
    main()