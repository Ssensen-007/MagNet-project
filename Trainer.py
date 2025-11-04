import torch
import numpy as np
import torch.nn as nn
from Model import Model
from utils import evaluate, evaluate_f1, log_results, EarlyStopping
from mctm_modules_graph import gate_entropy_loss, expert_balance_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from contrastive import InstanceContrastiveLoss, ClusterContrastiveLoss, generate_pseudo_labels


def build_icl_pairs(labels, augmented):
        # labels, augmented: numpy array
        orig_idx = np.where(augmented == 0)[0]
        aug_idx = np.where(augmented == 1)[0]
        # 假设每个原始样本的增强样本顺序一致（推荐你数据生成时保证顺序）
        # 可以直接按label和索引顺序配对
        label_to_aug_idx = {}
        for idx in aug_idx:
            l = labels[idx]
            label_to_aug_idx.setdefault(l, []).append(idx)
        pairs = []
        for idx in orig_idx:
            l = labels[idx]
            if l in label_to_aug_idx and len(label_to_aug_idx[l]) > 0:
                pairs.append((idx, label_to_aug_idx[l].pop(0)))
        orig_indices = np.array([p[0] for p in pairs])
        aug_indices = np.array([p[1] for p in pairs])
        return orig_indices, aug_indices

class Trainer:
    def __init__(self, params, h_dict, adj_dicts, labels, augmented, train_idx, val_idx, test_idx,
                 tfidf_word, tfidf_pos, tfidf_entity):
        self.params = params
        self.model = Model(params).to(params.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=1e-5)
        # ...其它初始化同原来...
        self.labels = labels.to(params.device)
        self.augmented = augmented.to(params.device)
        self.train_idx = train_idx.to(params.device)
        self.orig_indices, self.aug_indices = build_icl_pairs(
            self.labels.cpu().numpy(), self.augmented.cpu().numpy())
        
        if params.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=params.lr_scheduler_mode,
                factor=params.lr_scheduler_factor,
                patience=params.lr_scheduler_patience,
                min_lr=params.lr_scheduler_min_lr,
                threshold=params.lr_scheduler_threshold
            )
        else:
            self.scheduler = None

        self.h_dict = {k: v.to(params.device) for k, v in h_dict.items()}
        self.adj_dicts = {k: v.to(params.device) for k, v in adj_dicts.items()}
        self.labels = labels.to(params.device)
        self.train_idx = train_idx.to(params.device)
        self.val_idx = val_idx.to(params.device)
        self.test_idx = test_idx.to(params.device)
        self.tfidf_word = tfidf_word.to(params.device)
        self.tfidf_pos = tfidf_pos.to(params.device)
        self.tfidf_entity = tfidf_entity.to(params.device)

        # === 自定义标签损失函数设置 ===
        self.loss_mode = getattr(params, "loss_mode", "label_smooth")  # "cross_entropy" or "label_smooth"
        self.label_smooth = getattr(params, "label_smooth", 0.15)
        self.loss_fn = self.get_loss_fn(mode=self.loss_mode, smoothing=self.label_smooth)

        # === 标签分布统计（辅助分析）
        label_vals = [int(self.labels[i].cpu()) for i in train_idx]
        cnt = Counter(label_vals)
        print(f"[INFO] Train label distribution: {cnt}")

        print(f"标签分布: {torch.bincount(self.labels.cpu())}")
        print(f"train_idx 标签分布: {torch.bincount(self.labels[self.train_idx].cpu())}")
        print(f"val_idx 标签分布: {torch.bincount(self.labels[self.val_idx].cpu())}")

    def get_loss_fn(self, mode="label_smooth", smoothing=0.05):
        if mode == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif mode == "label_smooth":
            def label_smoothing_loss(pred, target):
                with torch.no_grad():
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(smoothing / (pred.size(1) - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)
                return torch.mean(torch.sum(-true_dist * torch.nn.functional.log_softmax(pred, dim=1), dim=1))
            return label_smoothing_loss
        else:
            raise ValueError(f"未知的损失函数模式: {mode}")

    def train(self):
        early_stopper = EarlyStopping(patience=self.params.early_patience, verbose=True)
        self.train_acc_log, self.val_acc_log, self.train_f1_log, self.val_f1_log = [], [], [], []
        best_val_acc = 0.0
        best_model = None 

        icl_loss_fn = InstanceContrastiveLoss(temperature=self.params.temperature).to(self.params.device)
        ccl_loss_fn = ClusterContrastiveLoss(temperature=self.params.temperature).to(self.params.device)
        for epoch in range(self.params.epochs):
            self.model.train()
            logits, doc_fea, proj_fea, gate_weights = self.model(
                self.h_dict, self.adj_dicts,
                self.tfidf_word, self.tfidf_pos, self.tfidf_entity,
                return_proj=True
            )
            # --- ICL损失 ---
            proj_orig = proj_fea[self.orig_indices]
            proj_aug  = proj_fea[self.aug_indices]
            icl_loss = icl_loss_fn(proj_orig, proj_aug)
            # --- CCL损失 ---
            all_train_proj = torch.cat([proj_orig, proj_aug], dim=0)
            with torch.no_grad():
                pseudo_labels = generate_pseudo_labels(all_train_proj, k=5)
            ccl_loss = ccl_loss_fn(all_train_proj, pseudo_labels)
            # --- 分类损失 ---
            ce_loss = nn.CrossEntropyLoss()(logits[self.train_idx], self.labels[self.train_idx])
            
            con_loss = self.params.icl_weight * icl_loss + self.params.ccl_weight * ccl_loss
            # --- MoE门控正则 ---
            moe_reg_loss = self.params.moe_entropy_weight * gate_entropy_loss(gate_weights) + self.params.moe_balance_weight * expert_balance_loss(gate_weights)
            total_loss = ce_loss + self.params.con_weight * con_loss + self.params.moe_weight * moe_reg_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()      
            # loss/grad检查
            if epoch % 10 == 0 or epoch < 3:
                print(f"[DEBUG] total_loss: {total_loss.item():.6f}")
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                print(f"[DEBUG] grad norm: {total_norm ** 0.5:.6f}")

            self.model.eval()
            with torch.no_grad():
                logits, doc_features = self.model(
                    self.h_dict, self.adj_dicts,
                    self.tfidf_word, self.tfidf_pos, self.tfidf_entity,
                    return_feats=True
                )
                if epoch % 10 == 0 or epoch < 3:
                    softmax_mean = torch.softmax(logits[self.val_idx], dim=1).mean(dim=0)
                    print(f"[DEBUG] val logits softmax mean: {softmax_mean}")
            
            train_acc = evaluate(logits, self.labels, self.train_idx)
            val_acc = evaluate(logits, self.labels, self.val_idx)
            train_f1_macro, _ = evaluate_f1(logits, self.labels, self.train_idx)
            val_f1_macro, _ = evaluate_f1(logits, self.labels, self.val_idx)
            print(f"[SanityCheck Epoch {epoch}] TrainAcc={train_acc:.4f}, TrainF1={train_f1_macro:.4f}")    
            if epoch % 10 == 0 or epoch < 3:
                preds = logits.argmax(dim=1)
                print(f"[DEBUG] epoch {epoch} val_pred_counts: {torch.bincount(preds[self.val_idx])}")
                print(f"[DEBUG] val_logits.softmax均值: {logits[self.val_idx].softmax(dim=1).mean(dim=0)}")
                print(f"[DEBUG] test_pred_counts: {torch.bincount(preds[self.test_idx])}")
            # === 行 127-130: 每轮预测分布、标签漂移、类别消失全检 ===
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                print(f"[DEBUG] 训练集预测类别分布: {torch.bincount(preds[self.train_idx].cpu())}")
                print(f"[DEBUG] 验证集预测类别分布: {torch.bincount(preds[self.val_idx].cpu())}")
                print(f"[DEBUG] 测试集预测类别分布: {torch.bincount(preds[self.test_idx].cpu())}")
            
            print("Train label counts:", torch.bincount(self.labels[self.train_idx].cpu()))
            print("Val label counts:", torch.bincount(self.labels[self.val_idx].cpu()))
            print("Test label counts:", torch.bincount(self.labels[self.test_idx].cpu()))

            self.train_acc_log.append(train_acc)
            self.val_acc_log.append(val_acc)
            self.train_f1_log.append(train_f1_macro)
            self.val_f1_log.append(val_f1_macro)
            print(f"[Epoch {epoch}] loss={total_loss:.4f} | TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f} | TrainF1={train_f1_macro:.4f}, ValF1={val_f1_macro:.4f}")

            early_stopper.step(val_acc)
            if early_stopper.early_stop:
                print(f"⏹️ 训练提前终止于第 {epoch} 轮（无提升）")
                print(f"✨ 当前最佳 val_acc: {best_val_acc:.4f}")
                break

            log_results(epoch, total_loss, train_acc, val_acc, train_f1_macro, val_f1_macro, file_path="training_log.csv")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = self.model.state_dict()
            
            if self.scheduler is not None:
                self.scheduler.step(val_acc)
            print('当前学习率:', self.optimizer.param_groups[0]['lr'])
        
        if best_model is not None:
            print("📌 加载最佳模型...")
            self.model.load_state_dict(best_model)
            # 输出最佳模型的 val_acc, val_f1
            with torch.no_grad():
                logits, _ = self.model(
                    self.h_dict, self.adj_dicts,
                    self.tfidf_word, self.tfidf_pos, self.tfidf_entity,
                    return_feats=True
                )
                val_acc = evaluate(logits, self.labels, self.val_idx)
                val_f1_macro, val_f1_micro = evaluate_f1(logits, self.labels, self.val_idx)
                print(f"🏆 Best Validation: val_acc={val_acc:.4f}, val_f1_macro={val_f1_macro:.4f}, val_f1_micro={val_f1_micro:.4f}")
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(
                self.h_dict, self.adj_dicts,
                self.tfidf_word, self.tfidf_pos, self.tfidf_entity,
                return_feats=True
            )
            acc = evaluate(logits, self.labels, self.test_idx)
            macro_f1, micro_f1 = evaluate_f1(logits, self.labels, self.test_idx)
            print(f"✅ [Test] Accuracy: {acc:.4f}")
            print(f"🎯 [Test] Macro-F1 : {macro_f1:.4f}")
            print(f"🎯 [Test] Micro-F1 : {micro_f1:.4f}")
