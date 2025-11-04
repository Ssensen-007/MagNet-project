class Params:
    def __init__(self):
        # === 基础训练参数 ===
        self.epochs = 500
        self.lr = 0.0002
        self.early_patience = 20  # 早停等待轮数
        self.seed = 42
        self.device = 'cuda:0'
        self.lr_scheduler = "plateau"      # 支持 "plateau" 或 "none"
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_threshold = 1e-4
        self.lr_scheduler_patience = 5
        self.lr_scheduler_min_lr = 1e-6
        self.lr_scheduler_mode = "max"

        # === 图结构表示维度 ===
        self.word_dim = 300    # GloVe词向量维度
        self.entity_dim = 300  # TransE实体向量维度
        self.pos_dim = 8       # POS标签one-hot维度
        self.hidden_dim = 128
        self.out_dim = 512  # 特征输出维度，所有节点类型建议统一
        self.dropout = 0.5
        self.in_dim = 300
        # === 损失函数权重 ===
        self.icl_weight = 0.5
        self.ccl_weight = 1.0 - self.icl_weight
        self.moe_entropy_weight = 0.3
        self.moe_balance_weight = 1.0 - self.moe_entropy_weight
        self.con_weight = 0.7
        self.moe_weight = 1.0 - self.con_weight
        # === 对比学习/MoE 相关参数 ===
        self.temperature = 0.5
        self.moe_temperature = 0.15 
        self.num_experts = 3      
        self.k = 2        
        # === 控制选项 ===
        self.loss_mode = "cross_entropy"    
        self.label_smooth = 0.05            

        # === 路径配置 ===
        self.save_model_path = "/root/autodl-tmp/zs/My_Model/best_model.pt"
        self.data_dir = "/root/autodl-tmp/zs/My_Model/data"
