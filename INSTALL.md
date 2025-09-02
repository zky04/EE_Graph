# 安装部署指南

## 快速部署

### 1. 环境准备
```bash
python --version
pip install -r requirements.txt
```

### 2. 数据准备
将数据目录重命名为 `dataset` 并移动到项目根目录下

### 3. 运行
```bash
python main.py --mode preprocessing
python main.py
```