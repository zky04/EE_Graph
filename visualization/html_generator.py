#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
class VisualizationGenerator:

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.get('data_paths.output_dir', '.'))
        self.viz_dir = Path(config.get('data_paths.visualization_dir', 'visualization'))
        self.viz_dir.mkdir(exist_ok=True)

    def generate_clustering_dashboard(self, clustering_results, topology_analysis=None):
        print("🎨 生成聚类分析仪表板...")

        if not clustering_results:
            print("❌ 聚类结果为空")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.viz_dir / f"clustering_dashboard_{timestamp}.html"

        html_content = self._create_simple_dashboard_html(clustering_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 交互式仪表板已生成: {output_file}")
        return str(output_file)

    def generate_all_visualizations(self, clustering_results, topology_analysis):
        return self.generate_clustering_dashboard(clustering_results, topology_analysis)

    def _generate_cluster_browser(self, clustering_results):
        try:
            html_content = f

            output_file = self.viz_dir / 'cluster_browser.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ 聚类浏览器已生成: {output_file}")
            return True

        except Exception as e:
            print(f"❌ 聚类浏览器生成失败: {e}")
            return False

    def _format_clusters_html(self, clusters, level):
        html_parts = []

        for cluster_id, cluster_info in list(clusters.items())[:10]:
            sample_count = cluster_info.get('sample_count', 0)

            html_parts.append(f)

        if len(clusters) > 10:
            html_parts.append(f'<p>... 还有 {len(clusters)-10} 个聚类</p>')

        return ''.join(html_parts)

    def _generate_topology_analyzer(self, clustering_results, topology_analysis):
        try:
            html_content = f

            output_file = self.viz_dir / 'topology_analyzer.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ 拓扑分析器已生成: {output_file}")
            return True

        except Exception as e:
            print(f"❌ 拓扑分析器生成失败: {e}")
            return False

    def _format_topology_table(self, level3_clusters):
        rows = []

        for cluster_id, cluster_info in list(level3_clusters.items())[:20]:
            sample_count = cluster_info.get('sample_count', 0)
            edge_analysis = cluster_info.get('edge_analysis', {})

            common_edges = edge_analysis.get('common_edge_count', 0)
            avg_unique = edge_analysis.get('avg_unique_edge_count', 0)
            score = edge_analysis.get('clustering_score', 0)

            rows.append(f)

        return ''.join(rows)

    def _generate_system_dashboard(self, clustering_results):
        try:
            total_timestamps = sum(len(cluster.get('timestamps', [])) for cluster in clustering_results.get('level3', {}).values())
            level1_count = len(clustering_results.get('level1', {}))
            level2_count = len(clustering_results.get('level2', {}))
            level3_count = len(clustering_results.get('level3', {}))

            html_content = f

            output_file = self.viz_dir / 'system_dashboard.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ 系统控制面板已生成: {output_file}")
            return True

        except Exception as e:
            print(f"❌ 系统控制面板生成失败: {e}")
            return False

    def _create_simple_dashboard_html(self, clustering_results):
        level1_count = len(clustering_results.get('level1_clusters', {}))
        level2_count = sum(len(clusters) for clusters in clustering_results.get('level2_clusters', {}).values())
        level3_count = sum(len(clusters) for clusters in clustering_results.get('level3_clusters', {}).values())

        html_template = f

        return html_template