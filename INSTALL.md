# 安装部署指南

## 方法1: 直接安装 (推荐)

```bash
git clone https://github.com/zky04/EE_Graph.git
cd EE_Graph
pip install -r requirements.txt
```

## 方法2: Conda环境

```bash
git clone https://github.com/zky04/EE_Graph.git
cd EE_Graph
conda env create -f environment.yml
conda activate ee_graph
```

## 方法3: 作为包安装

```bash
git clone https://github.com/zky04/EE_Graph.git
cd EE_Graph
pip install -e .
```

## 方法4: Docker部署

```bash
git clone https://github.com/zky04/EE_Graph.git
cd EE_Graph
docker-compose up --build
```

## 数据准备

将数据目录重命名为 `dataset` 并放在项目根目录

## 运行

```bash
python main.py --mode preprocessing
python main.py
```