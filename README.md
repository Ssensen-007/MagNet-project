# MagNet: Multi-view Adaptive Graph Network

**Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper **"Multi-view Adaptive Graph Network: Fusing Mixture of Experts with Dual-Contrastive Learning for Short-Text Classification"**.

MagNet is a novel framework designed to address the critical challenge of semantic sparsity in Short-Text Classification (STC). Instead of relying on static feature fusion (like concatenation or averaging), MagNet employs a dynamic, adaptive approach to combine information from multiple graph-based views.

## Core Contributions

1.  **Multi-view Graph Representation**: We construct a heterogeneous graph containing three views of information: lexical (word), syntactic (POS), and factual (entity). A `GCNEncoder` learns representations for each view.
2.  **Adaptive MoE Fusion**: We introduce a **Sparsely-gated Mixture of Experts (MoE) layer** (`MoEFeatureFusion`) to dynamically fuse the three view-specific document embeddings (`doc_word`, `doc_pos`, `doc_entity`). The gating network learns the optimal combination of experts for each individual text sample.
3.  **Dual-Granularity Contrastive Learning**: To enhance the discriminative power of the fused representations, we apply a dual-contrastive loss:
    * **Fine-Grained Instance Alignment (FIA)**: Pulls augmented pairs of the same text closer.
    * **Coarse-Grained Prototype Consistency (CPC)**: Enforces intra-cluster compactness based on generated pseudo-labels.
4.  **End-to-End Optimization**: The contrastive losses are applied to the final MoE output, allowing the supervisory signal to backpropagate through and optimize the entire network—including the MoE gate and the GCN encoders.

## Model Architecture

The data flow of MagNet is as follows:

1.  **Input:**
    * Corpus (View 1: Word, View 2: POS, View 3: Entity)
    * Graphs (Adjacency Matrices for each view)
    * Features (Embeddings for each view)
    * TF-IDF Matrices (for pooling)

2.  **Graph Encoding:**
    * The `GCNEncoder` processes the (Graphs + Features) for each of the three views.
    * **Output:** `Word Node Embs`, `POS Node Embs`, `Entity Node Embs`.

3.  **Pooling:**
    * Node Embs are multiplied with the TF-IDF Matrices.
    * **Output:** `doc_word`, `doc_pos`, `doc_entity` (document-level vectors for the three views).

4.  **Adaptive Fusion:**
    * `MoEFeatureFusion` (Mixture-of-Experts layer) receives the three document vectors.
    * `Sparse Gate` computes the weights.
    * `Experts` process each view.
    * **Output:** `doc_fea` (final fused document feature vector).

5.  **Multi-Task Output:**
    * **Branch 1 (Classification):** `doc_fea` -> Classifier -> `Logits` (for CE Loss).
    * **Branch 2 (Contrast):** `doc_fea` -> Projection Head -> `Proj Fea` (for FIA Loss and CPC Loss).
    * **Branch 3 (Regularization):** `Sparse Gate` -> MoE Regularization Losses (Entropy & Balance).

6.  **Total Loss:**
    * `Total Loss = CE Loss + Contrastive Loss + MoE Loss`.

## Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Ssensen-007/MagNet-project.git](https://github.com/Ssensen-007/MagNet-project.git)
    cd MagNet-project
    ```

2.  Install dependencies:
    (It is recommended to use a Conda virtual environment)
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

This model relies on pre-processed graphs, node features, and TF-IDF pooling matrices. Please place your datasets (e.g., `ag_news_data`) into the `data/` directory. The `train.py` script will load data from `BASE_DIR = "/root/autodl-tmp/zs/My_Model/data"` by default. You may need to modify this path.

**Important Note:** Due to GitHub's file size limit (over 100MB), you should not upload the pre-processed data files (`.pkl`) or raw datasets to this repository. You should use a `.gitignore` file to ignore them.

#### Download Datasets

You need to download the 6 benchmark datasets used in this study from the following public sources.

* **AG News:** [https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)
* **MR (Movie Review):** [https://www.cs.cornell.edu/people/pabo/movie-review-data/](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
* **Ohsumed:** (Common pre-processed version) [https://github.com/yao8839836/text_gcn/tree/master/data/ohsumed](https://github.com/yao8839836/text_gcn/tree/master/data/ohsumed) (Raw data: [https://trec.nist.gov/data/t9_filtering.html](https://trec.nist.gov/data/t9_filtering.html))
* **Twitter (Sentiment140):** [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Snippets:** [http://www.phanxuanhieu.com/source-code/](http://www.phanxuanhieu.com/source-code/)
* **TagMyNews:** [https://www.tagmynews.com/tagged_news_data.zip](https://www.tagmynews.com/tagged_news_data.zip)


Each dataset directory (e.g., `data/ag_news_data/`) must contain the following files:

* **Node Features:**
    * `word_emb.pkl`: Word node features (e.g., GloVe)
    * `pos_emb.pkl`: POS node features (e.g., one-hot)
    * `entity_emb.pkl`: Entity node features (e.g., TransE)
* **Adjacency Matrices:**
    * `adj_word2word.pkl`: Word-word adjacency matrix (Scipy COO format)
    * `adj_pos2pos.pkl`: POS-POS adjacency matrix
    * `adj_entity2entity.pkl`: Entity-entity adjacency matrix
* **Pooling Matrices:**
    * `tfidf_word.pkl`: Document-word TF-IDF matrix (Scipy COO format)
    * `tfidf_pos.pkl`: Document-POS pooling matrix
    * `tfidf_entity.pkl`: Document-entity pooling matrix
* **Labels and Indices:**
    * `labels.json`: Labels for all documents
    * `train_idx.json`: Training set indices
    * `val_idx.json`: Validation set indices
    * `test_idx.json`: Test set indices
    * `augmented_flags.json`: Flags indicating augmented (1) vs. original (0) samples

### 2. Configuration

All model hyperparameters, dimensions, and loss weights are defined in `params.py`. You can directly modify this file to adjust experimental settings.

Key parameters include:
* `hidden_dim`, `out_dim`: Dimensions for GCN and MoE
* `lr`, `epochs`, `early_patience`: Training parameters
* `con_weight`, `moe_weight`: Trade-off between contrastive loss vs. MoE regularization
* `FIA_weight`, `CPC_weight`: Trade-off between the two contrastive losses
* `temperature`: Temperature coefficient for contrastive learning
* `k`: Top-k sparse gating for MoE

### 3. Training

Running the `train.py` script directly will automatically train and evaluate all datasets defined in the `DATASETS` list.

/```bash
python train.py

### 4. Inference and Evaluation (Scoring) 

*  **Get Predictions and Labels:**
    * The function first gets the model's raw output (`logits`) and uses `torch.argmax(logits[index], dim=1)` to determine the predicted class (`preds`).
    * At the same time, it retrieves the corresponding true labels (true) for that data subset (e.g., `test_idx`).

*  **Calculate with scikit-learn:**
    * This code uses the `f1_score` function from the `scikit-learn` library to perform the professional F1 score calculation.
    * `preds` (predictions) and `true` (true labels) are passed to the `f1_score` function.

*  **Macro-F1:**
    * Calculated by setting `average='macro'`.。
    * **Calculation Process:** `scikit-learn` will independently calculate the F1 score for each class, and then compute the arithmetic mean of all class F1 scores. This metric treats all classes equally, regardless of whether a class has more or less data, making it very suitable for evaluating model performance on imbalanced datasets.
  
*  **Accuracy (ACC) :**
    * The function receives the model's raw scores (`logits`) and the true `labels` for a specific data split (e.g., `test_idx`).
    * It uses `torch.argmax(logits[index], dim=1)` to find the predicted class index (the one with the highest score) for each sample.
    * It compares these predictions (`preds`) to the true `labels (labels[index])`.
    * Score: The final score is `(Number of Correct Predictions) / (Total Number of Samples)`.

      
