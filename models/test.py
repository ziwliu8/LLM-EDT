
    # here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss, Balanced_Contrastive_Loss


class LLM4CDSR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num = self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        #pretrained_emb_count = llm_emb_all.shape[0]
        #emb_dim = llm_emb_all.shape[1]

        # !!! 你需要确定这个值，例如通过分析 inter_seq.pkl !!!
        # 假设 max_actual_item_id 是你数据里最大的物品ID
        #max_actual_item_id = 5593 # <--- 请替换成你数据中的实际最大ID

        # 确定嵌入层需要的大小（最大ID + 1 padding位）
        #embedding_size_needed = max_actual_item_id + 1

        #print(f"调试信息:")
        #print(f"A域物品数量: {self.item_numA}")
        #print(f"B域物品数量: {self.item_numB}")
        #print(f"预训练嵌入数量: {pretrained_emb_count}")
        #print(f"数据中最大物品ID: {max_actual_item_id}")
        #print(f"所需嵌入层大小: {embedding_size_needed}")

        # 创建最终的嵌入矩阵（初始化为0）
        #final_item_emb = np.zeros((embedding_size_needed, emb_dim))

        # 将预训练的嵌入填充进去（跳过索引0）
        # 确保不会超出预训练嵌入的范围
        #fill_count = min(pretrained_emb_count, embedding_size_needed - 1)
        #final_item_emb[1 : fill_count + 1, :] = llm_emb_all[:fill_count, :]
        #print(f"填充了 {fill_count} 个预训练嵌入。")

        # 如果需要的嵌入层大小 > (预训练数量 + 1 padding)，则需要随机初始化
        #if embedding_size_needed > pretrained_emb_count + 1:
        #    print(f"警告: 最大物品ID({max_actual_item_id})超出了预训练嵌入的数量({pretrained_emb_count})。")
        #    print(f"将为 ID 从 {pretrained_emb_count + 1} 到 {max_actual_item_id} 的物品创建随机嵌入。")
        #    num_random_init = embedding_size_needed - (pretrained_emb_count + 1)
        #    random_indices_start = pretrained_emb_count + 1
        #    final_item_emb[random_indices_start:, :] = np.random.normal(0, 0.02, (num_random_init, emb_dim))

        #print(f"最终物品嵌入矩阵形状: {final_item_emb.shape}")
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num = self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb

        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_item_emb = np.concatenate([
            np.zeros((1, llm_emb_all.shape[1])),
            llm_emb_all
        ])
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA + self.item_numB + 1, args.hidden_size, padding_idx=0)
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        if args.global_emb:  # if use the LLM embedding, do not initialize
            self.filter_init_modules.append("item_emb_llm")
        self._init_weights()

    def _get_embedding(self, log_seqs):
        item_seq_emb = self.item_emb_llm(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)
        return item_seq_emb

    def log2feats(self, log_seqs, positions):
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb_llm.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)
        return log_feats

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        pos_embs = self._get_embedding(pos)  # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg)  # (bs, max_seq_len, hidden_size)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)  # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss
    def predict(self, seq, item_indices, positions, **kwargs):  # for inference
        '''Used to predict the score of item_indices given log_seqs'''

        log_feats = self.log2feats(seq, positions)  # 移除 domain 参数
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices)  # 移除 domain 参数
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
    
class One4All(LLM4CDSR_base):

    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.domain_beta = args.domain_beta
        llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        
        # 检查用户嵌入的数量是否匹配
        total_users_in_emb = llm_user_emb.shape[0]
        if total_users_in_emb < self.user_num:
            print(f"警告: 预训练用户嵌入数量({total_users_in_emb})小于实际用户数量({self.user_num})")
            print(f"将为新增的用户创建随机初始化的嵌入向量")
            # 创建额外的随机嵌入
            emb_dim = llm_user_emb.shape[1]
            additional_users = self.user_num - total_users_in_emb
            additional_embeds = np.random.normal(0, 0.02, (additional_users, emb_dim))
            # 合并原有和新增的嵌入
            llm_user_emb = np.concatenate([llm_user_emb, additional_embeds], axis=0)
            
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        self.user_emb_llm.weight.requires_grad = False

        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
            nn.Linear(int(llm_user_emb.shape[1] / 2), args.hidden_size)
        )
        self.domain_interaction = nn.Sequential(
            nn.Linear(args.hidden_size , args.hidden_size//2),
            nn.ReLU(),
            nn.Linear(args.hidden_size//2, args.hidden_size),
            nn.LayerNorm(args.hidden_size)
        )
        self.balanced_contra_loss = Balanced_Contrastive_Loss(
            tau=args.tau,
            beta=args.domain_beta  # 在args中添加domain_beta参数
        )
        self.intra_domain_loss = nn.CosineSimilarity(dim=-1)
        self.hard_negative_weight = args.hard_negative_weight
# 获取LLM嵌入维度
        self.reg_loss_func  = Contrastive_Loss2(tau=args.tau_reg)
        self.user_loss_func = Contrastive_Loss2(tau=args.tau)
        self.filter_init_modules.append("user_emb_llm")
        self._init_weights()
        
    def get_hard_negatives(self, anchor, candidates, k=5):
        """选择最难的负样本
        Args:
            anchor: shape [batch_size, hidden_dim]
            candidates: shape [num_candidates, hidden_dim]
            k: 选择前k个最难的负样本
        Returns:
            hard_negatives: shape [batch_size, k, hidden_dim]
        """
        # 计算相似度矩阵 [batch_size, num_candidates]
        sim = torch.matmul(anchor, candidates.T)
    
        # 获取最相似的k个样本的索引 [batch_size, k]
        _, hard_negative_idx = sim.topk(k, dim=1)
    
        # 扩展索引维度以匹配gather操作
        # [batch_size, k] -> [batch_size, k, 1]
        hard_negative_idx = hard_negative_idx.unsqueeze(-1)
    
        # 扩展candidates以匹配gather操作
        # [num_candidates, hidden_dim] -> [batch_size, num_candidates, hidden_dim]
        candidates_expanded = candidates.unsqueeze(0).expand(anchor.size(0), -1, -1)
    
        # 使用gather收集硬负样本
        # [batch_size, k, hidden_dim]
        hard_negatives = torch.gather(
            candidates_expanded,
            1,
            hard_negative_idx.expand(-1, -1, candidates.shape[-1])
        )
    
        return hard_negatives
    
    def forward(self, seq, pos, neg, positions, reg_A, reg_B, user_id, **kwargs):
    # 1. 首先计算基础的序列推荐损失
        loss = super().forward(seq, pos, neg, positions, **kwargs)

    # 2. 获取所需的嵌入表示
        reg_A = reg_A[reg_A > 0]
        reg_B = reg_B[reg_B > 0]
        reg_A_emb = self._get_embedding(reg_A)
        reg_B_emb = self._get_embedding(reg_B)
        # 跨域交互
        reg_A_emb_enhanced = self.domain_interaction(reg_A_emb)
        reg_B_emb_enhanced = self.domain_interaction(reg_B_emb)
        
    # 3. 获取并处理用户LLM特征
        llm_feats = self.user_emb_llm(user_id)
        llm_feats = self.user_adapter(llm_feats)

        # 获取序列表征作为锚点
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]  # 使用最后一个时间步的表征

        # 计算域平衡的对比损失
        domain_balanced_loss = self.balanced_contra_loss(
            final_feat,
            reg_A_emb_enhanced,
            reg_B_emb_enhanced
        )
        loss += self.gamma * domain_balanced_loss

        hard_neg_A = self.get_hard_negatives(reg_A_emb, reg_A_emb)
        intra_A_loss = 0
        for i in range(hard_neg_A.shape[1]):
            neg_sample = hard_neg_A[:, i, :]
            pos_sim = self.intra_domain_loss(reg_A_emb, reg_A_emb)
            neg_sim = self.intra_domain_loss(reg_A_emb, neg_sample)
            intra_A_loss += torch.mean(torch.clamp(neg_sim - pos_sim + 0.1, min=0))
        intra_A_loss = intra_A_loss / hard_neg_A.shape[1]
        loss += self.hard_negative_weight * intra_A_loss

        hard_neg_B = self.get_hard_negatives(reg_B_emb, reg_B_emb)
        intra_B_loss = 0
        for i in range(hard_neg_B.shape[1]):
            neg_sample_B = hard_neg_B[:, i, :]
            pos_sim = self.intra_domain_loss(reg_B_emb, reg_B_emb)
            neg_sim = self.intra_domain_loss(reg_B_emb, neg_sample_B)
            intra_B_loss += torch.mean(torch.clamp(neg_sim - pos_sim + 0.1, min=0))
        intra_B_loss = intra_B_loss / hard_neg_B.shape[1]
        loss += self.hard_negative_weight * intra_B_loss
    #  计算正则化损失
        reg_loss = self.reg_loss_func(reg_A_emb_enhanced, reg_B_emb_enhanced)
        loss += self.alpha * reg_loss


    #  计算用户兴趣对齐损失
        user_loss = self.user_loss_func(llm_feats, final_feat)
        loss += self.beta * user_loss
    
    

        return loss