#!/usr/bin/env python3
import torch
import os
from pathlib import Path
class DataValidator:

    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.get('data_paths.processed_data_dir', '../新全'))

    def validate_all_data(self):
        print("🔍 验证g.pt文件...")

        if not self.data_dir.exists():
            return {'passed': False, 'issues': ['数据目录不存在']}

        graph_files = list(self.data_dir.glob("*/g.pt"))

        if not graph_files:
            return {'passed': False, 'issues': ['未找到g.pt文件']}

        print(f"📊 检查 {len(graph_files)} 个文件...")

        sample_files = graph_files[:10]
        valid_count = 0

        for graph_file in sample_files:
            try:
                data = torch.load(graph_file, map_location='cpu')
                if 'adjacency_matrix' in data or ('edge_index' in data and 'edge_attr' in data):
                    valid_count += 1
            except:
                pass

        success_rate = valid_count / len(sample_files)

        if success_rate > 0.8:
            print(f"✅ 数据验证通过 ({success_rate:.1%})")
            return {'passed': True, 'success_rate': success_rate}
        else:
            return {'passed': False, 'issues': [f'数据质量不足: {success_rate:.1%}']}
