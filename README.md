# EE图数据三层次聚类分析系统

基于电力系统图数据的智能聚类分析系统，实现从原始pandapower数据到可视化分析的完整流程。

## 快速开始

```bash
git clone https://github.com/zky04/EE_Graph.git
cd EE_Graph
pip install -r requirements.txt
python main.py --mode preprocessing
python main.py
```

## 安装方式

- **直接安装**: `pip install -r requirements.txt`
- **Conda环境**: `conda env create -f environment.yml`
- **作为包安装**: `pip install -e .`
- **Docker部署**: `docker-compose up --build`

详见 [INSTALL.md](INSTALL.md)

## 数据格式

原始数据：将数据目录重命名为 `dataset` 并放在项目根目录下
时间戳目录：`dataset/HU_YYYYMMDD_HHMM_mac_pp/new.txt`
处理后数据：`dataset/HU_YYYYMMDD_HHMM_mac_pp/graph.pt`
