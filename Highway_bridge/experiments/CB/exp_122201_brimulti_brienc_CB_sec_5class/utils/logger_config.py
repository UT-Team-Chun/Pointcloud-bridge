import logging
import sys
from pathlib import Path

def setup_logging(log_dir):
    """Setup logging with utf-8 encoding"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()  # 注意这里使用root logger
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create a file handler with utf-8 encoding
    file_handler = logging.FileHandler(log_dir / 'training.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a stream handler with utf-8 encoding
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# 全局变量存储logger实例
_logger = None

def get_logger():
    """获取全局logger实例"""
    global _logger
    if _logger is None:
        # 如果logger还没有设置，返回一个基础配置的logger
        _logger = logging.getLogger()
        if not _logger.handlers:  # 如果没有handler，添加一个基础的StreamHandler
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger

def initialize_logger(log_dir):
    """初始化全局logger"""
    global _logger
    _logger = setup_logging(log_dir)
    return _logger
