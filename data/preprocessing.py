#!/usr/bin/env python3
import os
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
class DataPreprocessor:

    def __init__(self, config):
        self.config = config

        preprocess_config = config.get('preprocessing', {})
        self.input_dir = preprocess_config.get('input_dir', 'dataset')
        self.output_dir = preprocess_config.get('output_dir', 'dataset')
        self.input_filename = preprocess_config.get('input_filename', 'new.txt')
        self.output_filename = preprocess_config.get('output_filename', 'graph.pt')

        self.feature_dims = preprocess_config.get('feature_dimensions', 5)
        self.encoding = preprocess_config.get('encoding', 'utf-8')

        defaults = preprocess_config.get('default_estimates', {})
        self.default_transformer_r = defaults.get('transformer_resistance', 0.01)
        self.default_transformer_x = defaults.get('transformer_reactance', 0.1)
        self.default_line_capacity = defaults.get('line_capacity_mva', 100.0)
        self.default_line_voltage = defaults.get('line_voltage_kv', 110.0)

        self.stats = {
            'total_transformers': 0,
            'total_lines': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'processing_errors': 0
        }

    def extract_transformer_parameters(self, text: str) -> List[Dict]:
        transformers = []

        transformer_pattern = r'pp\.create_transformer_from_parameters\(\s*net,\s*(.+?)\)'
        matches = re.findall(transformer_pattern, text, re.DOTALL)

        for match in matches:
            try:
                params = self._parse_parameters(match)
                if self._validate_transformer_params(params):
                    transformers.append(params)
                    self.stats['total_transformers'] += 1
            except Exception as e:
                logging.warning(f"解析transformer参数失败: {e}")
                self.stats['processing_errors'] += 1

        return transformers
    def extract_line_parameters(self, text: str) -> List[Dict]:
        lines = []

        line_pattern = r'pp\.create_line_from_parameters\(\s*net,\s*(.+?)\)'
        matches = re.findall(line_pattern, text, re.DOTALL)

        for match in matches:
            try:
                params = self._parse_parameters(match)
                if self._validate_line_params(params):
                    lines.append(params)
                    self.stats['total_lines'] += 1
            except Exception as e:
                logging.warning(f"解析line参数失败: {e}")
                self.stats['processing_errors'] += 1

        return lines
    def extract_generator_parameters(self, text: str) -> List[Dict]:
        generators = []

        if 'gen_count' not in self.stats:
            self.stats['gen_count'] = 0
            self.stats['ext_grid_count'] = 0
            self.stats['sgen_count'] = 0

        gen_pattern = r'pp\.create_gen\([^)]+\)'
        gen_matches = re.findall(gen_pattern, text, re.DOTALL)

        for i, gen_def in enumerate(gen_matches):
            gen_info = self._parse_generator_definition(gen_def, 'gen', i)
            if gen_info:
                generators.append(gen_info)
                self.stats['gen_count'] += 1

        ext_grid_pattern = r'pp\.create_ext_grid\([^)]+\)'
        ext_grid_matches = re.findall(ext_grid_pattern, text, re.DOTALL)

        for i, ext_grid_def in enumerate(ext_grid_matches):
            gen_info = self._parse_generator_definition(ext_grid_def, 'ext_grid', i)
            if gen_info:
                generators.append(gen_info)
                self.stats['ext_grid_count'] += 1

        sgen_pattern = r'pp\.create_sgen\([^)]+\)'
        sgen_matches = re.findall(sgen_pattern, text, re.DOTALL)

        for i, sgen_def in enumerate(sgen_matches):
            gen_info = self._parse_generator_definition(sgen_def, 'sgen', i)
            if gen_info:
                generators.append(gen_info)
                self.stats['sgen_count'] += 1

        self.stats['total_generators'] = len(generators)
        return generators

    def _parse_generator_definition(self, gen_def, gen_type, index):
        try:
            bus_match = re.search(r'bus=bus_(\d+)', gen_def)
            if not bus_match:
                return None

            bus_id = int(bus_match.group(1))

            params = {
                'bus': bus_id,
                'type': gen_type,
                'index': index
            }

            param_patterns = {
                'p_mw': r'p_mw=([0-9.-]+)',
                'q_mvar': r'q_mvar=([0-9.-]+)',
                'vm_pu': r'vm_pu=([0-9.]+)',
                'min_p_mw': r'min_p_mw=([0-9.]+)',
                'max_p_mw': r'max_p_mw=([0-9.]+)',
                'name': r'name="([^"]*)"'
            }

            for param_name, pattern in param_patterns.items():
                match = re.search(pattern, gen_def)
                if match:
                    if param_name == 'name':
                        params[param_name] = match.group(1)
                    else:
                        try:
                            params[param_name] = float(match.group(1))
                        except ValueError:
                            continue

            if 'name' not in params:
                params['name'] = f"{gen_type}_{index}"

            params['definition'] = gen_def.strip()

            return params

        except Exception as e:
            logging.warning(f"解析发电机定义失败: {e}")
            return None
    def _parse_parameters(self, param_string: str) -> Dict:
        params = {}

        patterns = {
            'hv_bus': r'hv_bus\s*=\s*bus_(\d+)',
            'lv_bus': r'lv_bus\s*=\s*bus_(\d+)',
            'from_bus': r'from_bus\s*=\s*bus_(\d+)',
            'bus': r'bus\s*=\s*bus_(\d+)',
            'to_bus': r'to_bus\s*=\s*bus_(\d+)',
            'sn_mva': r'sn_mva\s*=\s*([\d.]+)',
            'vn_hv_kv': r'vn_hv_kv\s*=\s*([\d.]+)',
            'vn_lv_kv': r'vn_lv_kv\s*=\s*([\d.]+)',
            'r_ohm_per_km': r'r_ohm_per_km\s*=\s*([\d.]+)',
            'x_ohm_per_km': r'x_ohm_per_km\s*=\s*([\d.]+)',
            'length_km': r'length_km\s*=\s*([\d.]+)',
            'p_mw': r'p_mw\s*=\s*([\d.-]+)',
            'vm_pu': r'vm_pu\s*=\s*([\d.]+)',
            'name': r"name\s*=\s*['\"]([^'\"]+)['\"]",
            'type': r"type\s*=\s*['\"]([^'\"]+)['\"]"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, param_string)
            if match:
                try:
                    if key in ['hv_bus', 'lv_bus', 'from_bus', 'to_bus', 'bus']:
                        params[key] = int(match.group(1))
                    elif key in ['name', 'type']:
                        params[key] = match.group(1)
                    else:
                        params[key] = float(match.group(1))
                except ValueError:
                    continue

        return params
    def _validate_transformer_params(self, params: Dict) -> bool:
        required = ['hv_bus', 'lv_bus', 'sn_mva', 'vn_hv_kv', 'vn_lv_kv']
        return all(key in params for key in required)
    def _validate_line_params(self, params: Dict) -> bool:
        required = ['from_bus', 'to_bus', 'r_ohm_per_km', 'x_ohm_per_km']
        return all(key in params for key in required)
    def _validate_generator_params(self, params: Dict) -> bool:
        required = ['bus']
        return all(key in params for key in required)
    def create_graph_data(self, transformers: List[Dict], lines: List[Dict], generators: List[Dict] = None) -> Dict:
        nodes = set()
        edges = []
        edge_features = []

        for transformer in transformers:
            hv_bus = transformer['hv_bus']
            lv_bus = transformer['lv_bus']

            nodes.add(hv_bus)
            nodes.add(lv_bus)

            edges.extend([[hv_bus, lv_bus], [lv_bus, hv_bus]])

            feature = [
                transformer['sn_mva'],
                transformer['vn_hv_kv'],
                transformer['vn_lv_kv'],
                self.default_transformer_r,
                self.default_transformer_x
            ]
            edge_features.extend([feature, feature])

        for line in lines:
            from_bus = line['from_bus']
            to_bus = line['to_bus']

            nodes.add(from_bus)
            nodes.add(to_bus)

            edges.extend([[from_bus, to_bus], [to_bus, from_bus]])

            feature = [
                self.default_line_capacity,
                self.default_line_voltage,
                self.default_line_voltage,
                line['r_ohm_per_km'],
                line['x_ohm_per_km']
            ]
            edge_features.extend([feature, feature])

        node_list = sorted(list(nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        edge_index = []
        for edge in edges:
            src_idx = node_to_idx[edge[0]]
            dst_idx = node_to_idx[edge[1]]
            edge_index.append([src_idx, dst_idx])

        graph_data = {
            'num_nodes': len(nodes),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(edge_features, dtype=torch.float),
            'node_mapping': node_to_idx,
            'original_nodes': node_list
        }

        if generators:
            node_generators = defaultdict(list)
            generator_info = {
                'total_count': len(generators),
                'by_type': {
                    'gen': self.stats.get('gen_count', 0),
                    'ext_grid': self.stats.get('ext_grid_count', 0),
                    'sgen': self.stats.get('sgen_count', 0)
                },
                'by_node': {},
                'all_generators': []
            }

            for gen in generators:
                bus_id = gen['bus']
                if bus_id in nodes:
                    node_generators[bus_id].append(gen)
                    generator_info['all_generators'].append(gen)

            generator_info['by_node'] = dict(node_generators)

            node_gen_features = []
            for node in node_list:
                if node in node_generators:
                    gens = node_generators[node]
                    total_p_mw = sum(g.get('p_mw', 0) for g in gens if g.get('p_mw') is not None)
                    total_q_mvar = sum(g.get('q_mvar', 0) for g in gens if g.get('q_mvar') is not None)
                    gen_count = len(gens)
                    has_ext_grid = any(g['type'] == 'ext_grid' for g in gens)
                    max_vm_pu = max(g.get('vm_pu', 1.0) for g in gens)

                    node_gen_features.append([
                        gen_count,
                        total_p_mw,
                        total_q_mvar,
                        1.0 if has_ext_grid else 0.0,
                        max_vm_pu
                    ])
                else:
                    node_gen_features.append([0.0, 0.0, 0.0, 0.0, 1.0])

            graph_data['generator_info'] = generator_info
            graph_data['node_generator_features'] = torch.tensor(node_gen_features, dtype=torch.float)

        self.stats['total_nodes'] = len(nodes)
        self.stats['total_edges'] = len(edges)

        return graph_data
    def process_timestamp_directory(self, timestamp_dir):
        input_path = Path(timestamp_dir) / self.input_filename
        output_path = Path(timestamp_dir) / self.output_filename

        if not input_path.exists():
            print(f"⚠️  跳过目录 {timestamp_dir}: 未找到 {self.input_filename}")
            return False

        try:
            with open(input_path, 'r', encoding=self.encoding) as f:
                content = f.read()

            transformers = self.extract_transformer_parameters(content)
            lines = self.extract_line_parameters(content)
            generators = self.extract_generator_parameters(content)

            if not transformers and not lines:
                print(f"⚠️  目录 {timestamp_dir}: 未找到有效的transformer或line参数")
                return False

            graph_data = self.create_graph_data(transformers, lines, generators)

            torch.save(graph_data, output_path)

            timestamp = os.path.basename(timestamp_dir)
            gen_info = f", 发电机={self.stats.get('total_generators', 0)} (gen:{self.stats.get('gen_count', 0)}, ext_grid:{self.stats.get('ext_grid_count', 0)}, sgen:{self.stats.get('sgen_count', 0)})" if self.stats.get('total_generators', 0) > 0 else ""
            print(f"✅ 时间戳 {timestamp}: Transformers={self.stats['total_transformers']}, Lines={self.stats['total_lines']}, 节点={self.stats['total_nodes']}, 边={self.stats['total_edges']//2}{gen_info}")

            return True

        except Exception as e:
            print(f"❌ 目录 {timestamp_dir} 处理失败: {e}")
            return False
    def process_all_data(self):
        input_dir = Path(self.input_dir)

        if not input_dir.exists():
            print(f"❌ 数据目录不存在: {input_dir}")
            return False

        timestamp_dirs = []
        for item in os.listdir(input_dir):
            item_path = input_dir / item
            if item_path.is_dir() and re.match(r'HU_\d{8}_\d{4}_mac_pp', item):
                timestamp_dirs.append(item_path)

        if not timestamp_dirs:
            print(f"❌ 在 {input_dir} 中未找到时间戳格式的子目录 (格式: HU_YYYYMMDD_HHMM_mac_pp)")
            return False

        print(f"🔍 找到 {len(timestamp_dirs)} 个时间戳目录")

        success_count = 0
        for timestamp_dir in tqdm(timestamp_dirs, desc="处理时间戳目录"):
            self.stats = {
                'total_transformers': 0,
                'total_lines': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'total_generators': 0,
                'gen_count': 0,
                'ext_grid_count': 0,
                'sgen_count': 0,
                'processing_errors': 0
            }

            if self.process_timestamp_directory(timestamp_dir):
                success_count += 1

        print(f"\n📊 处理完成: {success_count}/{len(timestamp_dirs)} 个目录成功处理")
        return success_count > 0