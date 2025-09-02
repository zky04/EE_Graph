#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_manager import ConfigManager
class HierarchicalClusteringSystem:

    def __init__(self, config):
        if isinstance(config, str):
            self.config = ConfigManager(config)
        else:
            self.config = config

        data_dir = self.config.get('data_paths.processed_data_dir', '../新全')
        self.data_dir = Path(data_dir)

        self.clusters = {"level1": {}, "level2": {}, "level3": {}}
        self.timestamps = []

        if self.config.get('performance.enable_graph_cache', True):
            self.graph_cache = {}
        else:
            self.graph_cache = None

        print("🚀 修复后的三层次聚类系统初始化完成")
        if self.config:
            self.config.print_config_summary()
        self._load_timestamps()

    def _load_timestamps(self):
        print("📁 加载时间戳列表...")

        all_timestamps = [
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "g.pt").exists()
        ]

        if self.config.get('debug.sample_mode', False):
            sample_size = self.config.get('debug.sample_size', 100)
            self.timestamps = sorted(all_timestamps)[:sample_size]
            print(f"🧪 样本模式: 使用 {len(self.timestamps)} 个时间戳 (共 {len(all_timestamps)} 个)")
        else:
            self.timestamps = all_timestamps
            print(f"✅ 加载完成: {len(self.timestamps)} 个时间戳")

    def load_graph_data(self, timestamp):
        if self.graph_cache is not None and timestamp in self.graph_cache:
            return self.graph_cache[timestamp]

        graph_path = self.data_dir / timestamp / "g.pt"
        if not graph_path.exists():
            return None

        try:
            graph_data = torch.load(graph_path, map_location='cpu')
            if self.graph_cache is not None:
                self.graph_cache[timestamp] = graph_data
            return graph_data
        except Exception as e:
            print(f"加载 {timestamp} 失败: {e}")
            return None

    def extract_level1_features(self, graph_data):
        if 'adjacency_matrix' in graph_data:
            adj_matrix = graph_data['adjacency_matrix']
            if hasattr(adj_matrix, 'shape'):
                n_nodes = adj_matrix.shape[0]
            else:
                raise ValueError(f"Adjacency matrix type {type(adj_matrix)} does not have shape attribute")
        elif 'num_nodes' in graph_data:
            n_nodes = graph_data['num_nodes']
        elif 'edge_index' in graph_data:
            edge_index = graph_data['edge_index']
            n_nodes = edge_index.max().item() + 1
        else:
            raise KeyError("Graph data missing node count information (adjacency_matrix, num_nodes, or edge_index)")

        features = {
            'bus_count': n_nodes,
            'bus_group': self._categorize_bus_count(n_nodes)
        }

        return features

    def _categorize_bus_count(self, bus_count):
        categories = self.config.get('level1.bus_categories', {
            'small': 100, 'medium': 300, 'large': 600, 'extra_large': 999999
        })

        enable_fine_grouping = self.config.get('level1.enable_fine_grouping', True)
        fine_group_size = self.config.get('level1.fine_group_size', 3)

        if enable_fine_grouping and bus_count >= 600:
            if bus_count <= 617:
                return "group_0"
            elif bus_count <= 620:
                return "group_1"
            elif bus_count <= 623:
                return "group_2"
            elif bus_count <= 626:
                return "group_3"
            elif bus_count <= 629:
                return "group_4"
            elif bus_count <= 632:
                return "group_5"
            elif bus_count <= 635:
                return "group_6"
            elif bus_count <= 638:
                return "group_7"
            elif bus_count <= 641:
                return "group_8"
            else:
                return "group_9"
        else:
            if bus_count < categories['small']:
                return "small"
            elif bus_count < categories['medium']:
                return "medium"
            elif bus_count < categories['large']:
                return "large"
            else:
                return "extra_large"

    def perform_level1_clustering(self):
        print("🔵 Level 1聚类: 按Bus数量分组...")

        bus_groups = defaultdict(list)

        for timestamp in tqdm(self.timestamps, desc="Level 1特征提取"):
            graph_data = self.load_graph_data(timestamp)
            if graph_data:
                features = self.extract_level1_features(graph_data)
                bus_group = features['bus_group']
                bus_groups[bus_group].append({
                    'timestamp': timestamp,
                    'bus_count': features['bus_count']
                })

        cluster_id = 0
        for group_name, group_data in bus_groups.items():
            self.clusters["level1"][cluster_id] = {
                'cluster_id': f"level1_{cluster_id}",
                'bus_group': group_name,
                'timestamps': [item['timestamp'] for item in group_data],
                'sample_count': len(group_data),
                'bus_count_range': (
                    min(item['bus_count'] for item in group_data),
                    max(item['bus_count'] for item in group_data)
                )
            }
            cluster_id += 1

        print(f"✅ Level 1聚类完成: {len(self.clusters['level1'])} 个聚类")
        return self.clusters["level1"]

    def extract_bus_generator_pattern(self, graph_data):
        if 'generator_info' not in graph_data:
            return {
                'total_generators': 0,
                'generator_types': {},
                'bus_generator_distribution': {},
                'generator_capacity_pattern': {},
                'signature': "no_generators"
            }

        gen_info = graph_data['generator_info']

        total_generators = gen_info.get('total_count', 0)
        generator_types = gen_info.get('by_type', {})

        bus_generator_distribution = {}
        generator_capacity_pattern = {}

        by_node = gen_info.get('by_node', {})

        for bus_id, generators in by_node.items():
            bus_id = int(bus_id)

            bus_gen_count = len(generators)
            bus_gen_types = {}
            bus_total_capacity = 0

            for gen in generators:
                gen_type = gen.get('type', 'unknown')
                bus_gen_types[gen_type] = bus_gen_types.get(gen_type, 0) + 1

                p_mw = gen.get('p_mw', 0)
                if p_mw is not None:
                    bus_total_capacity += p_mw

            bus_generator_distribution[bus_id] = {
                'count': bus_gen_count,
                'types': bus_gen_types,
                'capacity': bus_total_capacity
            }

            capacity_range = self._categorize_capacity(bus_total_capacity)
            generator_capacity_pattern[bus_id] = capacity_range

        signature = self._create_bus_generator_signature(
            total_generators, generator_types, bus_generator_distribution
        )

        return {
            'total_generators': total_generators,
            'generator_types': generator_types,
            'bus_generator_distribution': bus_generator_distribution,
            'generator_capacity_pattern': generator_capacity_pattern,
            'signature': signature,
            'buses_with_generators': len(bus_generator_distribution),
            'avg_generators_per_bus': total_generators / len(bus_generator_distribution) if bus_generator_distribution else 0
        }

    def _categorize_capacity(self, capacity):
        if capacity == 0:
            return "no_capacity"
        elif capacity <= 100:
            return "small_0_100MW"
        elif capacity <= 300:
            return "medium_100_300MW"
        elif capacity <= 600:
            return "large_300_600MW"
        elif capacity <= 1000:
            return "very_large_600_1000MW"
        else:
            return "huge_1000MW_plus"

    def _create_bus_generator_signature(self, total_generators, generator_types, bus_distribution):
        sig_parts = [f"total_{total_generators}"]

        type_sig = []
        for gen_type, count in sorted(generator_types.items()):
            type_sig.append(f"{gen_type}:{count}")
        if type_sig:
            sig_parts.append("types_" + "_".join(type_sig))

        buses_with_gen = len(bus_distribution)
        sig_parts.append(f"buses_{buses_with_gen}")

        capacity_categories = {}
        for bus_id, info in bus_distribution.items():
            capacity_range = self._categorize_capacity(info['capacity'])
            capacity_categories[capacity_range] = capacity_categories.get(capacity_range, 0) + 1

        capacity_sig = []
        for capacity_range, count in sorted(capacity_categories.items()):
            capacity_sig.append(f"{capacity_range}:{count}")
        if capacity_sig:
            sig_parts.append("capacity_" + "_".join(capacity_sig))

        return "|".join(sig_parts)

    def extract_graph_structure_pattern(self, graph_data):
        import torch

        edge_index = graph_data['edge_index']
        edge_attr = graph_data['edge_attr']

        num_nodes = edge_index.max().item() + 1
        num_edges = edge_index.shape[1]

        structure_signature = f"exact_n{num_nodes}_e{num_edges}"

        return {
            'generator_count': num_nodes,
            'connection_counts': [num_edges],
            'unique_connected_buses': [num_nodes, num_edges],
            'connection_signature': structure_signature,
            'avg_connections_per_gen': num_edges / num_nodes if num_nodes > 0 else 0,
            'exact_features': {
                'num_nodes': num_nodes,
                'num_edges': num_edges
            }
        }

    def _create_connection_signature(self, generator_connections):
        patterns = []

        for gen_id, info in generator_connections.items():
            connected = sorted(info['connected_buses'][:5])
            pattern = f"{info['connection_count']}:{','.join(map(str, connected))}"
            patterns.append(pattern)

        return "|".join(sorted(patterns))

    def calculate_bus_generator_similarity(self, pattern1, pattern2):
        if pattern1['signature'] == pattern2['signature']:
            return 1.0

        total_gen_sim = 0.0
        if pattern1['total_generators'] == pattern2['total_generators']:
            total_gen_sim = 1.0
        elif abs(pattern1['total_generators'] - pattern2['total_generators']) <= 2:
            total_gen_sim = 0.8
        elif abs(pattern1['total_generators'] - pattern2['total_generators']) <= 5:
            total_gen_sim = 0.5
        else:
            total_gen_sim = 0.0

        types1 = pattern1['generator_types']
        types2 = pattern2['generator_types']
        type_sim = self._calculate_type_distribution_similarity(types1, types2)

        buses1 = pattern1['buses_with_generators']
        buses2 = pattern2['buses_with_generators']
        bus_count_sim = 0.0
        if buses1 == buses2:
            bus_count_sim = 1.0
        elif abs(buses1 - buses2) <= 1:
            bus_count_sim = 0.8
        elif abs(buses1 - buses2) <= 2:
            bus_count_sim = 0.5

        capacity_sim = self._calculate_capacity_pattern_similarity(
            pattern1['generator_capacity_pattern'],
            pattern2['generator_capacity_pattern']
        )

        weights = self.config.get('level2.similarity_weights', {
            'total_generators': 0.3,
            'generator_types': 0.3,
            'bus_count': 0.2,
            'capacity_pattern': 0.2
        })

        total_similarity = (
            weights['total_generators'] * total_gen_sim +
            weights['generator_types'] * type_sim +
            weights['bus_count'] * bus_count_sim +
            weights['capacity_pattern'] * capacity_sim
        )

        return total_similarity

    def _calculate_type_distribution_similarity(self, types1, types2):
        if not types1 and not types2:
            return 1.0
        if not types1 or not types2:
            return 0.0

        all_types = set(types1.keys()) | set(types2.keys())
        similarities = []

        for gen_type in all_types:
            count1 = types1.get(gen_type, 0)
            count2 = types2.get(gen_type, 0)

            if count1 == count2:
                similarities.append(1.0)
            elif abs(count1 - count2) == 1:
                similarities.append(0.8)
            elif abs(count1 - count2) <= 2:
                similarities.append(0.5)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_capacity_pattern_similarity(self, pattern1, pattern2):
        if not pattern1 and not pattern2:
            return 1.0
        if not pattern1 or not pattern2:
            return 0.0

        capacity_count1 = {}
        capacity_count2 = {}

        for capacity_range in pattern1.values():
            capacity_count1[capacity_range] = capacity_count1.get(capacity_range, 0) + 1

        for capacity_range in pattern2.values():
            capacity_count2[capacity_range] = capacity_count2.get(capacity_range, 0) + 1

        all_ranges = set(capacity_count1.keys()) | set(capacity_count2.keys())
        similarities = []

        for capacity_range in all_ranges:
            count1 = capacity_count1.get(capacity_range, 0)
            count2 = capacity_count2.get(capacity_range, 0)

            if count1 == count2:
                similarities.append(1.0)
            elif abs(count1 - count2) == 1:
                similarities.append(0.8)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0
    def calculate_graph_structure_similarity(self, pattern1, pattern2):
        if 'exact_features' in pattern1 and 'exact_features' in pattern2:
            nodes_match = pattern1['exact_features']['num_nodes'] == pattern2['exact_features']['num_nodes']
            edges_match = pattern1['exact_features']['num_edges'] == pattern2['exact_features']['num_edges']

            if nodes_match and edges_match:
                return 1.0
            else:
                return 0.0

        return 1.0 if pattern1['connection_signature'] == pattern2['connection_signature'] else 0.0

    def perform_level2_clustering(self, similarity_threshold=None):
        print("🟡 Level 2聚类: 基于bus上发电机分布分析...")

        if similarity_threshold is None:
            similarity_threshold = self.config.get('level2.similarity_threshold', 0.8)

        print(f"   相似度阈值: {similarity_threshold}")
        cluster_id_counter = 0

        for level1_cluster in self.clusters["level1"].values():
            timestamps = level1_cluster['timestamps']

            if len(timestamps) <= 1:
                self.clusters["level2"][cluster_id_counter] = {
                    'cluster_id': f"level2_{cluster_id_counter}",
                    'parent_level1': level1_cluster['cluster_id'],
                    'timestamps': timestamps,
                    'sample_count': len(timestamps)
                }
                cluster_id_counter += 1
                continue

            generator_patterns = {}
            for timestamp in tqdm(timestamps, desc=f"Level2分析 {level1_cluster['cluster_id']}"):
                graph_data = self.load_graph_data(timestamp)
                if graph_data:
                    pattern = self.extract_bus_generator_pattern(graph_data)
                    generator_patterns[timestamp] = pattern

            level2_subclusters = self._cluster_by_bus_generator_similarity(
                generator_patterns, similarity_threshold
            )

            for subcluster_timestamps in level2_subclusters:
                self.clusters["level2"][cluster_id_counter] = {
                    'cluster_id': f"level2_{cluster_id_counter}",
                    'parent_level1': level1_cluster['cluster_id'],
                    'timestamps': subcluster_timestamps,
                    'sample_count': len(subcluster_timestamps)
                }
                cluster_id_counter += 1

        print(f"✅ Level 2聚类完成: {len(self.clusters['level2'])} 个聚类")
        return self.clusters["level2"]

    def _cluster_by_bus_generator_similarity(self, generator_patterns, threshold):
        timestamps = list(generator_patterns.keys())

        if len(timestamps) <= 1:
            return [timestamps]

        n = len(timestamps)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                ts1, ts2 = timestamps[i], timestamps[j]
                similarity = self.calculate_bus_generator_similarity(
                    generator_patterns[ts1], generator_patterns[ts2]
                )
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

        clusters = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i+1, n):
                if j not in used and similarity_matrix[i, j] >= threshold:
                    cluster.append(j)
                    used.add(j)

            cluster_timestamps = [timestamps[idx] for idx in cluster]
            clusters.append(cluster_timestamps)

        for i in range(n):
            if i not in used:
                clusters.append([timestamps[i]])

        return clusters

    def extract_edge_set_with_features(self, graph_data):
        if 'edge_index' in graph_data and 'edge_attr' in graph_data:
            edge_index = graph_data['edge_index']
            edge_attr = graph_data['edge_attr']
        else:
            return self.extract_edge_set_legacy(graph_data)

        edge_features = {}

        processed_edges = set()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i].tolist()
            edge_key = tuple(sorted([src, dst]))

            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)

            features = edge_attr[i].tolist()

            edge_type = self.classify_edge_type(features)
            feature_signature = self.create_edge_signature(features, edge_type)

            edge_features[edge_key] = {
                'features': features,
                'type': edge_type,
                'feature_only_signature': feature_signature
            }

        return edge_features

    def classify_edge_type(self, features):
        sn_mva, vn_hv_kv, vn_lv_kv, r_ohm, x_ohm = features

        if r_ohm < 0.1 and x_ohm < 1.0:
            return "transformer"
        else:
            return "line"

    def create_edge_signature(self, features, edge_type):
        sn_mva, vn_hv_kv, vn_lv_kv, r_ohm, x_ohm = features

        if edge_type == "transformer":
            return f"{sn_mva:.1f}_{vn_hv_kv:.1f}_{vn_lv_kv:.1f}_{r_ohm:.3f}_{x_ohm:.3f}"
        else:
            return f"{sn_mva:.1f}_{vn_hv_kv:.1f}_{vn_lv_kv:.1f}_{r_ohm:.4f}_{x_ohm:.4f}"

    def extract_edge_set_legacy(self, graph_data):
        if 'adjacency_matrix' not in graph_data:
            return {}

        adj_matrix = graph_data['adjacency_matrix']

        if hasattr(adj_matrix, 'numpy'):
            adj_matrix = adj_matrix.numpy()

        edge_features = {}

        for i in range(adj_matrix.shape[0]):
            for j in range(i+1, adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    edge_key = (i, j)
                    default_features = [100.0, 110.0, 10.5, 0.01, 0.1]
                    edge_type = self.classify_edge_type(default_features)
                    feature_signature = self.create_edge_signature(default_features, edge_type)

                    edge_features[edge_key] = {
                        'features': default_features,
                        'edge_type': edge_type,
                        'feature_only_signature': feature_signature,
                        'full_signature': f"{edge_key}_{feature_signature}"
                    }

        return edge_features

    def analyze_common_and_unique_edges(self, timestamps):
        if len(timestamps) <= 1:
            return {
                'common_edges': set(),
                'unique_edges': {},
                'total_edges': set(),
                'edge_analysis': {}
            }

        edge_features_dict = {}
        for timestamp in timestamps:
            graph_data = self.load_graph_data(timestamp)
            if graph_data:
                edge_features_dict[timestamp] = self.extract_edge_set_with_features(graph_data)

        if not edge_features_dict:
            return {'common_edges': set(), 'unique_edges': {}, 'total_edges': set()}

        common_edges = {}
        unique_edges = {}
        all_signatures = set()

        feature_signature_to_timestamps = defaultdict(set)
        for timestamp, edge_features in edge_features_dict.items():
            unique_edges[timestamp] = set()
            for edge_key, edge_info in edge_features.items():
                feature_signature = edge_info['feature_only_signature']
                all_signatures.add(feature_signature)
                feature_signature_to_timestamps[feature_signature].add(timestamp)

        num_graphs = len(edge_features_dict)
        for feature_signature, timestamp_set in feature_signature_to_timestamps.items():
            if len(timestamp_set) == num_graphs:
                common_edges[feature_signature] = {
                    'feature_signature': feature_signature,
                    'graph_count': len(timestamp_set),
                    'timestamps': list(timestamp_set)
                }
            else:
                for timestamp in timestamp_set:
                    unique_edges[timestamp].add(feature_signature)

        edge_analysis = {
            'common_edge_count': len(common_edges),
            'total_edge_signatures': len(all_signatures),
            'common_edge_ratio': len(common_edges) / len(all_signatures) if all_signatures else 0,
            'unique_edge_counts': {ts: len(unique) for ts, unique in unique_edges.items()},
            'avg_unique_edge_count': np.mean([len(unique) for unique in unique_edges.values()]) if unique_edges else 0,
            'common_edge_details': {sig: info for sig, info in common_edges.items()},
            'graph_count': len(edge_features_dict)
        }

        return {
            'common_edges': common_edges,
            'unique_edges': unique_edges,
            'all_signatures': all_signatures,
            'edge_analysis': edge_analysis
        }

    def calculate_topology_clustering_score(self, edge_analysis):
        if not edge_analysis or edge_analysis['total_edge_signatures'] == 0:
            return 0.0

        common_ratio_score = edge_analysis['common_edge_ratio']

        avg_unique_count = edge_analysis['avg_unique_edge_count']
        total_edge_count = edge_analysis['total_edge_signatures']

        unique_edge_ratio = avg_unique_count / total_edge_count if total_edge_count > 0 else 0
        unique_penalty = max(0, 1.0 - unique_edge_ratio)

        common_weight = self.config.get('level3.scoring.common_edge_weight', 0.7)
        unique_weight = self.config.get('level3.scoring.unique_edge_penalty', 0.3)

        clustering_score = common_weight * common_ratio_score + unique_weight * unique_penalty

        return clustering_score

    def perform_level3_clustering(self, min_common_edges=None, max_unique_edges=None):
        print("🔴 Level 3聚类: 拓扑结构分析(共同边/独特边)...")

        if min_common_edges is None:
            min_common_edges = self.config.get('level3.common_edges.min_count', 5)
        if max_unique_edges is None:
            max_unique_edges = self.config.get('level3.unique_edges.max_count', 20)

        print(f"   最少共同边: {min_common_edges}")
        print(f"   最多独特边: {max_unique_edges}")

        cluster_id_counter = 0

        for level2_cluster in tqdm(self.clusters["level2"].values(), desc="Level 3拓扑分析"):
            timestamps = level2_cluster['timestamps']

            if len(timestamps) <= 1:
                self.clusters["level3"][cluster_id_counter] = {
                    'cluster_id': f"{cluster_id_counter}_0_0",
                    'parent_level2': level2_cluster['cluster_id'],
                    'timestamps': timestamps,
                    'sample_count': len(timestamps),
                    'edge_analysis': {
                        'common_edge_count': 0,
                        'avg_unique_edge_count': 0,
                        'common_edge_ratio': 1.0,
                        'clustering_score': 1.0
                    }
                }
                cluster_id_counter += 1
                continue

            level3_subclusters = self._cluster_by_topology_similarity(
                timestamps, min_common_edges, max_unique_edges
            )

            for subcluster_timestamps in level3_subclusters:
                edge_analysis_result = self.analyze_common_and_unique_edges(subcluster_timestamps)
                edge_stats = edge_analysis_result.get('edge_analysis', {})
                clustering_score = self.calculate_topology_clustering_score(edge_stats)

                self.clusters["level3"][cluster_id_counter] = {
                    'cluster_id': f"{cluster_id_counter}_0_0",
                    'parent_level2': level2_cluster['cluster_id'],
                    'timestamps': subcluster_timestamps,
                    'sample_count': len(subcluster_timestamps),
                    'edge_analysis': {
                        'common_edge_count': edge_stats.get('common_edge_count', 0),
                        'avg_unique_edge_count': edge_stats.get('avg_unique_edge_count', 0),
                        'common_edge_ratio': edge_stats.get('common_edge_ratio', 0),
                        'clustering_score': clustering_score
                    }
                }
                cluster_id_counter += 1

        print(f"✅ Level 3聚类完成: {len(self.clusters['level3'])} 个聚类")
        return self.clusters["level3"]

    def _cluster_by_topology_similarity(self, timestamps, min_common_edges, max_unique_edges):
        if len(timestamps) <= 1:
            return [timestamps]

        edge_features_dict = {}
        for timestamp in timestamps:
            graph_data = self.load_graph_data(timestamp)
            if graph_data:
                edge_features_dict[timestamp] = self.extract_edge_set_with_features(graph_data)

        clusters = []
        remaining = list(timestamps)

        while remaining:
            seed = remaining[0]
            seed_signatures = set()
            if seed in edge_features_dict:
                seed_signatures = {info['feature_only_signature'] for info in edge_features_dict[seed].values()}
            current_cluster = [seed]
            remaining.remove(seed)

            to_remove = []
            for candidate in remaining:
                if candidate not in edge_features_dict:
                    continue

                candidate_signatures = {info['feature_only_signature'] for info in edge_features_dict[candidate].values()}

                common_signatures = seed_signatures.intersection(candidate_signatures)
                unique_signatures_seed = seed_signatures - common_signatures
                unique_signatures_candidate = candidate_signatures - common_signatures

                meets_criteria = (
                    len(common_signatures) >= min_common_edges and
                    len(unique_signatures_seed) <= max_unique_edges and
                    len(unique_signatures_candidate) <= max_unique_edges
                )

                if meets_criteria:
                    current_cluster.append(candidate)
                    to_remove.append(candidate)
                    seed_signatures = seed_signatures.intersection(candidate_signatures)

            for node in to_remove:
                remaining.remove(node)

            clusters.append(current_cluster)

        return clusters

    def run_complete_clustering(self):
        print("🚀 开始修复后的三层次聚类分析...")
        print("=" * 60)

        self.perform_level1_clustering()

        self.perform_level2_clustering()

        self.perform_level3_clustering()

        self._generate_clustering_summary()

        print("🎉 三层次聚类分析完成!")
        return self.clusters

    def _generate_clustering_summary(self):
        summary = {
            'total_timestamps': len(self.timestamps),
            'level1_clusters': len(self.clusters["level1"]),
            'level2_clusters': len(self.clusters["level2"]),
            'level3_clusters': len(self.clusters["level3"]),
            'clustering_logic': {
                'level1': 'Bus数量聚类',
                'level2': '发电机连接相似度聚类',
                'level3': '拓扑结构聚类(共同边/独特边分析)'
            }
        }

        output_dir = Path(self.config.get('data_paths.output_dir', '.'))
        summary_filename = self.config.get('output.summary_filename', 'clustering_summary')
        results_filename = self.config.get('output.results_filename', 'clustering_results')

        output_dir.mkdir(exist_ok=True)

        level1_dir = output_dir / 'level1'
        level2_dir = output_dir / 'level2'
        level3_dir = output_dir / 'level3'

        level1_dir.mkdir(exist_ok=True)
        level2_dir.mkdir(exist_ok=True)
        level3_dir.mkdir(exist_ok=True)

        summary_path = output_dir / f"{summary_filename}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        results_path = output_dir / f"{results_filename}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            serializable_clusters = {}
            for level, clusters in self.clusters.items():
                serializable_clusters[level] = {}
                for cid, cluster in clusters.items():
                    cluster_copy = cluster.copy()
                    if 'edge_analysis' in cluster_copy:
                        edge_analysis = cluster_copy['edge_analysis']
                        if isinstance(edge_analysis, dict):
                            for key, value in edge_analysis.items():
                                if isinstance(value, set):
                                    cluster_copy['edge_analysis'][key] = list(value)
                    serializable_clusters[level][cid] = cluster_copy

            json.dump(serializable_clusters, f, ensure_ascii=False, indent=2)

        level_dirs = {
            'level1': level1_dir,
            'level2': level2_dir,
            'level3': level3_dir
        }

        saved_files = []
        for level, clusters in self.clusters.items():
            if level in level_dirs:
                level_dir = level_dirs[level]

                level_results = {}
                for cid, cluster in clusters.items():
                    cluster_copy = cluster.copy()
                    if 'edge_analysis' in cluster_copy:
                        edge_analysis = cluster_copy['edge_analysis']
                        if isinstance(edge_analysis, dict):
                            for key, value in edge_analysis.items():
                                if isinstance(value, set):
                                    cluster_copy['edge_analysis'][key] = list(value)
                    level_results[cid] = cluster_copy

                level_file = level_dir / f"{level}_results.json"
                with open(level_file, 'w', encoding='utf-8') as f:
                    json.dump(level_results, f, ensure_ascii=False, indent=2)
                saved_files.append(level_file)

        print("📊 聚类汇总:")
        print(f"  Level 1 (Bus数量): {summary['level1_clusters']} 个聚类")
        print(f"  Level 2 (发电机连接): {summary['level2_clusters']} 个聚类")
        print(f"  Level 3 (拓扑结构): {summary['level3_clusters']} 个聚类")
        print(f"  覆盖时间戳: {summary['total_timestamps']} 个")
        print(f"\n📁 结果保存:")
        print(f"  汇总报告: {summary_path}")
        print(f"  详细结果: {results_path}")
        print(f"  分级结果:")
        for saved_file in saved_files:
            print(f"    {saved_file}")
def main():
    print("🔧 运行修复后的三层次聚类系统")
    print("=" * 60)
    print("聚类逻辑:")
    print("  Level 1: Bus数量聚类")
    print("  Level 2: 发电机连接相似度聚类")
    print("  Level 3: 拓扑结构聚类(共同边/独特边分析)")
    print("=" * 60)

    clustering_system = HierarchicalClusteringSystem("config.yaml")

    results = clustering_system.run_complete_clustering()

    print("\n📋 修复完成! 结果文件:")
    print("  - fixed_clustering_summary.json: 聚类汇总")
    print("  - fixed_cluster_results.json: 详细聚类结果")
if __name__ == "__main__":
    main()