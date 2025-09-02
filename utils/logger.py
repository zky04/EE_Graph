#!/usr/bin/env python3
import logging
import os
from pathlib import Path
from datetime import datetime
def setup_logger(config):

    log_dir = Path(config.get('data_paths.logs_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)

    log_filename = config.get('output.log_filename', 'clustering_analysis')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if config.get('output.timestamp_suffix', True):
        log_file = log_dir / f"{log_filename}_{timestamp}.log"
    else:
        log_file = log_dir / f"{log_filename}.log"

    level_str = config.get('debug.log_level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('EEGraphAnalysis')
    logger.setLevel(level)
    logger.addHandler(file_handler)

    if config.get('debug.verbose_logging', False):
        logger.addHandler(console_handler)

    logger.info("日志系统初始化完成")
    return logger