import os
import torch
import json
import pandas as pd
import argparse
from params import Params
from utils import load_all_data, set_seed, evaluate, evaluate_f1, load_tfidf_matrices
from Trainer import Trainer

DATASETS = [
    "ag_news_data",
    "mr_data",
    "ohsumed_data",
    "twitter_data",
    "snippets_data",
    "tagmynews_data"
]
BASE_DIR = "/root/autodl-tmp/zs/My_Model/data" 

summary_results = []

def run_dataset(dataset_name):
    print(f"\n🚀 正在训练数据集：{dataset_name}")
    params = Params()
    set_seed(params.seed)
    params.data_dir = os.path.join(BASE_DIR, dataset_name)
    params.save_model_path = os.path.join(params.data_dir, "best_model.pt")
    
    # # ==== 加载最佳超参 ====
    # best_param_path = f"best_params_{dataset_name}.json"
    # if os.path.exists(best_param_path):
    #     with open(best_param_path, "r") as f:
    #         best_params = json.load(f)
    #     for k, v in best_params.items():
    #         setattr(params, k, v)
    #     print(f"✅ 已加载最优参数: {best_param_path}")
    # else:
    #     print(f"⚠️ 未找到 {best_param_path}，使用默认参数。")

    # ✅ 加载数据
    print("📦 加载图结构与特征数据...")

    # === MODIFIED: 只加载word/pos/entity特征和同构邻接 ===
    h_dict, adj_dicts, labels, train_idx, val_idx, test_idx, augmented = load_all_data(params)
    
    # === 新增: 加载TF-IDF（或池化权重）矩阵 ===
    tfidf_word, tfidf_pos, tfidf_entity = load_tfidf_matrices(params.data_dir)
    print("✅ 数据加载完毕")

    params.num_classes = len(set(labels.tolist()))

    print("word_emb shape:", h_dict[0].shape)
    print("pos_emb shape:", h_dict[1].shape)
    print("entity_emb shape:", h_dict[2].shape)

    # === MODIFIED: Trainer和模型forward要接收tfidf池化矩阵 ===
    trainer = Trainer(params, h_dict, adj_dicts, labels, augmented, train_idx, val_idx, test_idx,
    tfidf_word, tfidf_pos, tfidf_entity)
    trainer.train()
    trainer.test()

    # ✅ 保存模型
    torch.save(trainer.model.state_dict(), params.save_model_path)
    print(f"💾 最佳模型已保存为 {params.save_model_path}")

    # ✅ 保存训练日志
    try:
        metrics = {
            'epoch': list(range(len(trainer.train_acc_log))),
            'train_acc': trainer.train_acc_log,
            'val_acc': trainer.val_acc_log,
            'train_f1': trainer.train_f1_log,
            'val_f1': trainer.val_f1_log  
        }
        log_path = os.path.join(params.data_dir, "train_log.csv")
        df = pd.DataFrame(metrics)
        df.to_csv(log_path, index=False)
        print(f"📈 日志保存为 {log_path}")
    except Exception as e:
        print("⚠️ 日志保存失败（可忽略）：", e)

    # ✅ 保存验证集 & 测试集 F1 分数
    try:
        val_logits, _ = trainer.model(
            h_dict, adj_dicts, tfidf_word, tfidf_pos, tfidf_entity, return_feats=True
        )
        val_f1_macro, val_f1_micro = evaluate_f1(val_logits, labels, val_idx)
        val_acc = evaluate(val_logits, labels, val_idx)
        
        test_logits, _ = trainer.model(
            h_dict, adj_dicts, tfidf_word, tfidf_pos, tfidf_entity, return_feats=True
        )
        test_f1_macro, test_f1_micro = evaluate_f1(test_logits, labels, test_idx)
        test_acc = evaluate(test_logits, labels, test_idx)
        
        # 新增更详细的预测分布/置信度
        if hasattr(val_logits, "softmax"):
            with torch.no_grad():
                probs = torch.softmax(val_logits, dim=1)
                print("[DEBUG] 验证集 logits softmax 分布前5:", probs[:5].detach().cpu().numpy())
                # [约行 115] 检查各集预测分布
                print("[DEBUG] val_pred_counts:", torch.argmax(val_logits, dim=1)[val_idx].bincount())
                print("[DEBUG] test_pred_counts:", torch.argmax(test_logits, dim=1)[test_idx].bincount())
        
        summary = {
            "dataset": dataset_name,
            "val_acc": round(val_acc, 4),
            "val_f1_macro": round(val_f1_macro, 4),
            "val_f1_micro": round(val_f1_micro, 4),
            "test_acc": round(test_acc, 4),
            "test_f1_macro": round(test_f1_macro, 4),
            "test_f1_micro": round(test_f1_micro, 4)
        }

        # 保存单个数据集 JSON
        summary_path = os.path.join(params.data_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"📊 F1 结果保存为 {summary_path}")

        # 加入全局 summary_results 列表
        summary_results.append(summary)

    except Exception as e:
        print("⚠️ F1 评估指标保存失败（可忽略）：", e)
    

if __name__ == "__main__":
    for ds in DATASETS:
        run_dataset(ds)

    # ✅ 保存所有数据集汇总的 summary.csv
    try:
        df = pd.DataFrame(summary_results)
        df.to_csv("summary_all_results.csv", index=False)
        print("📊 所有数据集汇总 F1 已保存为 summary_all_results.csv")
    except Exception as e:
        print("⚠️ 汇总表保存失败（可忽略）：", e)
