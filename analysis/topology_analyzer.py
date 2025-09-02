#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
class TopologyAnalyzer:

    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.get('data_paths.processed_data_dir', '../新全'))

    def analyze_all_clusters(self, clustering_results):
        print("🔬 分析聚类拓扑特征...")

        if not clustering_results or 'level3' not in clustering_results:
            print("❌ 未找到Level 3聚类结果")
            return None

        analysis_results = {}
        level3_clusters = clustering_results['level3']

        print(f"📊 分析 {len(level3_clusters)} 个Level 3聚类...")

        for cluster_id, cluster_info in level3_clusters.items():
            timestamps = cluster_info['timestamps']

            topo_analysis = self._analyze_cluster_topology(timestamps, cluster_id)
            analysis_results[cluster_id] = topo_analysis

        print("✅ 拓扑分析完成")

        output_file = Path(self.config.get('data_paths.output_dir', '.')) / "topology_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        return analysis_results

    def _analyze_cluster_topology(self, timestamps, cluster_id):
        if not timestamps:
            return {'error': '空聚类'}

        sample_timestamp = timestamps[0]
        graph_path = self.data_dir / sample_timestamp / "graph.pt"

        try:
            data = torch.load(graph_path, map_location='cpu')
            adj_matrix = data['adjacency_matrix'].numpy()

            analysis = {
                'cluster_id': cluster_id,
                'sample_count': len(timestamps),
                'representative_timestamp': sample_timestamp,
                'nodes': adj_matrix.shape[0],
                'edges': int(np.sum(adj_matrix > 0) // 2),
                'density': np.sum(adj_matrix > 0) / (adj_matrix.shape[0] * (adj_matrix.shape[0] - 1))
            }

            return analysis

        except Exception as e:
            return {'error': str(e)}