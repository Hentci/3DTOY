import logging
import sys
from typing import Optional

def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """設置日誌記錄器

    Args:
        name (str): 日誌記錄器名稱
        level (str, optional): 日誌級別. 預設是 'INFO'
        log_file (str, optional): 日誌文件路徑. 預設是 None

    Returns:
        logging.Logger: 配置好的日誌記錄器
    """
    # 創建日誌記錄器
    logger = logging.getLogger(name)
    
    # 如果已經有處理器，不重複添加
    if logger.handlers:
        return logger
        
    # 設置日誌格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 設置控制台輸出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日誌文件，添加文件處理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 設置日誌級別
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    return logger

# 測試代碼
if __name__ == "__main__":
    logger = setup_logger("test_logger", "DEBUG")
    logger.debug("這是一條調試訊息")
    logger.info("這是一條信息")
    logger.warning("這是一條警告")
    logger.error("這是一條錯誤")