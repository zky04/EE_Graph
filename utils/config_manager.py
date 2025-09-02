#!/usr/bin/env python3
import yaml
import os
from pathlib import Path
class ConfigManager:

    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        self.load_config()

    def load_config(self):
        if not self.config_path.exists():
            print(f"❌ 配置文件 {self.config_path} 不存在")
            self.create_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            self.config = self.get_default_config()

    def get_default_config(self):
        return {
            'data_paths': {
                'input_dir': '../新全',
                'output_dir': '.',
                'logs_dir': 'logs',
                'reports_dir': 'reports'
            },
            'level1': {
                'bus_categories': {'small': 100, 'medium': 300, 'large': 600, 'extra_large': 999999},
                'enable_fine_grouping': True,
                'fine_group_size': 50
            },
            'level2': {
                'similarity_threshold': 0.8,
                'similarity_weights': {
                    'connection_signature': 0.5,
                    'generator_count': 0.2,
                    'bus_overlap': 0.2,
                    'avg_connections': 0.1
                }
            },
            'level3': {
                'common_edges': {'min_count': 5, 'min_ratio': 0.1},
                'unique_edges': {'max_count': 20, 'max_ratio': 0.1},
                'scoring': {'common_edge_weight': 0.7, 'unique_edge_penalty': 0.3}
            },
            'performance': {
                'enable_graph_cache': True,
                'show_progress_bar': True
            },
            'output': {
                'save_json': True,
                'generate_html': True,
                'detailed_reports': True
            }
        }

    def create_default_config(self):
        default_config = self.get_default_config()

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"✅ 创建默认配置文件: {self.config_path}")
            self.config = default_config
        except Exception as e:
            print(f"❌ 创建配置文件失败: {e}")
            self.config = default_config

    def get(self, key_path, default=None):
        if self.config is None:
            return default

        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path, value):
        if self.config is None:
            self.config = {}

        keys = key_path.split('.')
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"✅ 配置已保存: {self.config_path}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")

    def validate_config(self):
        print("🔍 验证配置文件...")

        required_keys = [
            'data_paths.input_dir',
            'level1.bus_categories',
            'level2.similarity_threshold',
            'level3.common_edges.min_count',
            'level3.unique_edges.max_count'
        ]

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            print(f"❌ 配置验证失败，缺少必需配置: {missing_keys}")
            return False
        else:
            print("✅ 配置验证通过")
            return True

    def print_config_summary(self):
        print("📋 当前配置摘要:")
        print(f"  数据目录: {self.get('data_paths.input_dir')}")
        print(f"  Level 1 - Bus分类阈值: {self.get('level1.bus_categories')}")
        print(f"  Level 2 - 相似度阈值: {self.get('level2.similarity_threshold')}")
        print(f"  Level 3 - 最少共同边: {self.get('level3.common_edges.min_count')}")
        print(f"  Level 3 - 最多独特边: {self.get('level3.unique_edges.max_count')}")
        print(f"  缓存启用: {self.get('performance.enable_graph_cache')}")
        print(f"  生成HTML: {self.get('output.generate_html')}")
def test_config_manager():
    print("🧪 测试配置管理器...")

    config = ConfigManager("config.yaml")

    config.validate_config()

    config.print_config_summary()

    print("\n🔧 测试配置读取:")
    print(f"Level 2相似度阈值: {config.get('level2.similarity_threshold')}")
    print(f"Level 3最少共同边: {config.get('level3.common_edges.min_count')}")
    print(f"Level 3最多独特边: {config.get('level3.unique_edges.max_count')}")

    return config
if __name__ == "__main__":
    test_config_manager()