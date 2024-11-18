import json

def load_config(file_path='config.json'):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

# 全局配置变量
CONFIG = load_config()