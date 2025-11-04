# MagNet: Multi-view Adaptive Graph Network

**Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper **"Multi-view Adaptive Graph Network: Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification"**.

MagNet is a novel framework designed to address the critical challenge of semantic sparsity in Short-Text Classification (STC). Instead of relying on static feature fusion (like concatenation or averaging), MagNet employs a dynamic, adaptive approach to combine information from multiple graph-based views.

## 核心贡献 (Core Contributions)

1.  **Multi-view Graph Representation**: We construct a heterogeneous graph containing three views of information: lexical (word-word), syntactic (POS-POS), and factual (entity-entity). A `GCNEncoder` learns representations for each view.
2.  **Adaptive MoE Fusion**: We introduce a **Sparsely-gated Mixture of Experts (MoE) layer** (`MoEFeatureFusion`) to dynamically fuse the three view-specific document embeddings (`doc_word`, `doc_pos`, `doc_entity`). The gating network learns the optimal combination of experts for each individual text sample.
3.  **Dual-Granularity Contrastive Learning**: To enhance the discriminative power of the fused representations, we apply a dual-contrastive loss:
    * **Instance-level (ICL)**: Pulls augmented pairs of the same text closer.
    * **Cluster-level (CCL)**: Enforces intra-cluster compactness based on generated pseudo-labels.
4.  **End-to-End Optimization**: The contrastive losses are applied to the final MoE output, allowing the supervisory signal to backpropagate through and optimize the entire network—including the MoE gate and the GCN encoders.

## 模型架构 (Model Architecture)
