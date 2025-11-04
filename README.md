# MagNet: Multi-view Adaptive Graph Network

**Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper **"Multi-view Adaptive Graph Network: Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification"**.

MagNet is a novel framework designed to address the critical challenge of semantic sparsity in Short-Text Classification (STC). Instead of relying on static feature fusion (like concatenation or averaging), MagNet employs a dynamic, adaptive approach to combine information from multiple graph-based views.

## 核心贡献 (Core Contributions)

1.  **Multi-view Graph Representation**: We construct a heterogeneous graph containing three views of information: lexical (word), syntactic (POS), and factual (entity). A `GCNEncoder` learns representations for each view.
2.  **Adaptive MoE Fusion**: We introduce a **Sparsely-gated Mixture of Experts (MoE) layer** (`MoEFeatureFusion`) to dynamically fuse the three view-specific document embeddings (`doc_word`, `doc_pos`, `doc_entity`). The gating network learns the optimal combination of experts for each individual text sample.
3.  **Dual-Granularity Contrastive Learning**: To enhance the discriminative power of the fused representations, we apply a dual-contrastive loss:
    * **Fine-Grained Instance Alignment (FIA)**: Pulls augmented pairs of the same text closer.
    * **Coarse-Grained Prototype Consistency (CPC)**: Enforces intra-cluster compactness based on generated pseudo-labels.
4.  **End-to-End Optimization**: The contrastive losses are applied to the final MoE output, allowing the supervisory signal to backpropagate through and optimize the entire network—including the MoE gate and the GCN encoders.

## 模型架构 (Model Architecture)

MagNet 的数据流如下：

1.  **输入 (Input):** * Corpus (View 1: Word, View 2: POS, View 3: Entity)
    * Graphs (Adjacency Matrices for each view)
    * Features (Embeddings for each view)
    * TF-IDF Matrices (for pooling)

2.  **图编码 (Graph Encoding):**
    * `GCNEncoder` 分别处理三个视图的 (Graphs + Features)。
    * **输出:** `Word Node Embs`, `POS Node Embs`, `Entity Node Embs`.

3.  **池化 (Pooling):**
    * Node Embs 与 TF-IDF Matrices 相乘。
    * **输出:** `doc_word`, `doc_pos`, `doc_entity` (三个视图的文档级向量).

4.  **自适应融合 (Adaptive Fusion):**
    * `MoEFeatureFusion` (专家混合层) 接收三个文档向量。
    * `Sparse Gate` (稀疏门控) 计算权重。
    * `Experts` (专家网络) 处理各视图。
    * **输出:** `doc_fea` (最终融合的文档特征向量).

5.  **多任务输出 (Multi-Task Output):**
    * **分支 1 (分类):** `doc_fea` -> Classifier -> `Logits` (用于 CE Loss).
    * **分支 2 (对比):** `doc_fea` -> Projection Head -> `Proj Fea` (用于 ICL Loss 和 CCL Loss).
    * **分支 3 (正则):** `Sparse Gate` -> MoE Regularization Losses (Entropy & Balance).

6.  **总损失 (Total Loss):**
    * `Total Loss = CE Loss + Contrastive Loss + MoE Loss`.

## 安装 (Installation)

1.  克隆本仓库:
    ```bash
    git clone [https://github.com/Ssensen-007/MagNet-project.git](https://github.com/Ssensen-007/MagNet-project.git)
    cd MagNet-project
    ```

2.  安装依赖:
    (建议使用 Conda 创建虚拟环境)
    ```bash
    pip install -r requirements.txt
    ```

## 如何运行 (Usage)

### 1. 数据准备 (Data Preparation)

本模型依赖于预处理好的图、节点特征和 TF-IDF 池化矩阵。请将您的数据集（例如 `ag_news_data`）放入 `data/` 目录中。`train.py` 脚本默认会从 `BASE_DIR = "/root/autodl-tmp/zs/My_Model/data"` 加载数据。您可能需要修改此路径。

**重要提示：** 由于 GitHub 限制大文件上传（超过 100MB），您不应将预处理好的数据文件（`.pkl`）或原始数据集上传到本仓库。您应该使用 `.gitignore` 文件来忽略它们。

#### 数据集下载 (Download Datasets)

您需要从以下公开来源下载本研究中使用的 6 个基准数据集。

* **AG News:** [https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)
* **MR (Movie Review):** [https://www.cs.cornell.edu/people/pabo/movie-review-data/](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
* **Ohsumed:** (常用预处理版本) [https://github.com/yao8839836/text_gcn/tree/master/data/ohsumed](https://github.com/yao8839836/text_gcn/tree/master/data/ohsumed) (原始数据: [https://trec.nist.gov/data/t9_filtering.html](https://trec.nist.gov/data/t9_filtering.html))
* **Twitter (Sentiment140):** [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Snippets:** [http://www.phanxuanhieu.com/source-code/](http://www.phanxuanhieu.com/source-code/)
* **TagMyNews:** [https://www.tagmynews.com/tagged_news_data.zip](https://www.tagmynews.com/tagged_news_data.zip)


每个数据集目录（例如 `data/ag_news_data/`）必须包含以下文件：

* **节点特征 (Node Features):**
    * `word_emb.pkl`: 词节点特征 (e.g., GloVe)
    * `pos_emb.pkl`: POS 节点特征 (e.g., one-hot)
    * `entity_emb.pkl`: 实体节点特征 (e.g., TransE)
* **邻接矩阵 (Adjacency Matrices):**
    * `adj_word2word.pkl`: 词-词邻接矩阵 (Scipy COO format)
    * `adj_pos2pos.pkl`: POS-POS 邻接矩阵
    * `adj_entity2entity.pkl`: 实体-实体邻接矩阵
* **池化矩阵 (Pooling Matrices):**
    * `tfidf_word.pkl`: 文档-词 TF-IDF 矩阵 (Scipy COO format)
    * `tfidf_pos.pkl`: 文档-POS 池化矩阵
    * `tfidf_entity.pkl`: 文档-实体 池化矩阵
* **标签和索引 (Labels and Indices):**
    * `labels.json`: 所有文档的标签
    * `train_idx.json`: 训练集索引
    * `val_idx.json`: 验证集索引
    * `test_idx.json`: 测试集索引
    * `augmented_flags.json`: 标记哪些是增强样本 (1) vs 原始样本 (0)

### 2. 参数配置 (Configuration)

所有模型的超参数、维度和损失权重都在 `params.py` 文件中定义。您可以直接修改此文件来调整实验设置。

关键参数包括：
* `hidden_dim`, `out_dim`: GCN 和 MoE 的维度
* `lr`, `epochs`, `early_patience`: 训练参数
* `con_weight`, `moe_weight`: 损失函数（对比学习 vs MoE正则）的权衡
* `icl_weight`, `ccl_weight`: 两种对比学习损失内部的权衡
* `temperature`: 对比学习的温度系数
* `k`: MoE 的 Top-k 稀疏门控

### 3. 训练模型 (Training)

直接运行 `train.py` 脚本将自动训练和评估 `DATASETS` 列表中定义的所有数据集。

```bash
python train.py
