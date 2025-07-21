# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

# Can be placed in models/utils.py or models/SASRec.py, or even at the top of domain_specific_adapter.py
# For organization, let's assume it's added to models/utils.py (or a new file like models/attention.py)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        #self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        #self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        #self.relu = torch.nn.ReLU()
        #self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        #self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        self.w_1 = nn.Linear(hidden_units, hidden_units//2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(hidden_units//2, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, inputs):
        #outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        #outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        #outputs += inputs
        outputs = self.w_1(inputs)
        outputs = self.dropout1(outputs)
        outputs = self.activation(outputs)
        outputs = self.w_2(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.LayerNorm(outputs + inputs)
        return outputs


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y, reduction='mean'):
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss # Return per-sample loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    
def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return torch.sum(F.softplus(neg_preds - pos_preds))


class IntraDomainLoss(nn.Module):
    def __init__(self, margin=0.1, hard_negative_weight=0.1):
        super().__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        
    def forward(self, domain_A_emb, domain_B_emb):
        """
        计算域内对比损失
        
        Args:
            domain_A_emb: [num_A, hidden_dim] A域物品表征
            domain_B_emb: [num_B, hidden_dim] B域物品表征
        """
        # 为保证安全，首先检查是否有足够的样本
        if domain_A_emb.size(0) < 2 or domain_B_emb.size(0) < 2:
            return torch.tensor(0.0, device=domain_A_emb.device)
            
        # 标准化嵌入
        domain_A_emb = F.normalize(domain_A_emb, dim=1)
        domain_B_emb = F.normalize(domain_B_emb, dim=1)
        
        # 计算A域内的相似度矩阵
        sim_A = torch.matmul(domain_A_emb, domain_A_emb.t())
        # 创建目标，对角线为1，其他为0（自己和自己相似）
        identity_A = torch.eye(domain_A_emb.size(0), device=domain_A_emb.device)
        
        # 计算A域内的硬负例损失（难分样本）
        hard_negatives_A = torch.where(sim_A > self.margin, sim_A, torch.zeros_like(sim_A))
        hard_negatives_A = hard_negatives_A * (1 - identity_A)  # 移除对角线元素
        hard_negative_loss_A = (hard_negatives_A.sum() / (domain_A_emb.size(0) - 1)) if domain_A_emb.size(0) > 1 else 0
        
        # 计算B域内的相似度矩阵
        sim_B = torch.matmul(domain_B_emb, domain_B_emb.t())
        # 创建目标，对角线为1，其他为0
        identity_B = torch.eye(domain_B_emb.size(0), device=domain_B_emb.device)
        
        # 计算B域内的硬负例损失
        hard_negatives_B = torch.where(sim_B > self.margin, sim_B, torch.zeros_like(sim_B))
        hard_negatives_B = hard_negatives_B * (1 - identity_B)  # 移除对角线元素
        hard_negative_loss_B = (hard_negatives_B.sum() / (domain_B_emb.size(0) - 1)) if domain_B_emb.size(0) > 1 else 0
        
        # 总损失 = A域内损失 + B域内损失 + 硬负例权重 * (A域硬负例损失 + B域硬负例损失)
        loss = hard_negative_loss_A + hard_negative_loss_B
        
        return self.hard_negative_weight * loss

class Balanced_Contrastive_Loss(nn.Module):
    def __init__(self, tau=0.07, beta=2.0):
        super().__init__()
        self.tau = tau
        self.beta = beta
        
    def forward(self, anchor, domain_A_emb, domain_B_emb):
        """
        计算平衡的对比损失
        
        Args:
            anchor: [batch_size, hidden_dim] 锚点表征（用户全局兴趣）
            domain_A_emb: [num_A, hidden_dim] A域物品表征
            domain_B_emb: [num_B, hidden_dim] B域物品表征
        """
        # 为保证安全，首先检查是否有足够的样本
        if domain_A_emb.size(0) < 2 or domain_B_emb.size(0) < 2:
            return torch.tensor(0.0, device=anchor.device)
            
        # 标准化嵌入
        anchor = F.normalize(anchor, dim=1)
        domain_A_emb = F.normalize(domain_A_emb, dim=1)
        domain_B_emb = F.normalize(domain_B_emb, dim=1)
        
        # 计算A域损失
        logits_A = torch.matmul(anchor, domain_A_emb.t()) / self.tau
        labels_A = torch.zeros(logits_A.size(0), dtype=torch.long, device=anchor.device)
        loss_A = F.cross_entropy(logits_A, labels_A)
        
        # 计算B域损失
        logits_B = torch.matmul(anchor, domain_B_emb.t()) / self.tau
        labels_B = torch.zeros(logits_B.size(0), dtype=torch.long, device=anchor.device)
        loss_B = F.cross_entropy(logits_B, labels_B)
        
        # 平衡两个域的损失
        balanced_loss = loss_A + self.beta * loss_B
        
        return balanced_loss

class CrossDomainInterestLoss(nn.Module):
    # Restoring original implementation with hard negative mining
    def __init__(self, tau=0.05, hard_negative_weight=0.5, margin=0.3):
        super().__init__()
        self.tau = tau  # 温度参数
        self.hard_negative_weight = hard_negative_weight  # 难负样本权重
        self.margin = margin  # 相似度阈值，用于选择难负样本
        
    def _calculate_hnm_loss(self, sim_matrix, margin):
        """
        Calculates hard negative mining loss for a given similarity matrix.
        Assumes sim_matrix is [bs, bs] where sim_matrix[i,i] is positive-like.
        Hard negatives are off-diagonal elements > margin.
        """
        bs = sim_matrix.size(0)
        if bs <= 1: # Need more than one item to have negatives
            return torch.tensor(0.0, device=sim_matrix.device)

        user_hard_loss_sums = []
        for i in range(bs):
            neg_scores_for_user_i = []
            for j in range(bs):
                if i == j:
                    continue
                neg_scores_for_user_i.append(sim_matrix[i, j])
            
            if not neg_scores_for_user_i:
                continue

            neg_scores_tensor = torch.stack(neg_scores_for_user_i)
            hard_negatives = neg_scores_tensor[neg_scores_tensor > margin]

            if hard_negatives.numel() > 0:
                user_hard_loss_sums.append(hard_negatives.sum())
        
        if not user_hard_loss_sums:
            return torch.tensor(0.0, device=sim_matrix.device)
        
        return torch.stack(user_hard_loss_sums).mean()
        
    def forward(self, user_interest, reg_A_emb, reg_B_emb):
        """
        计算跨域兴趣对比损失 (原始实现)
        
        Args:
            user_interest: [batch_size, hidden_dim] 用户整体兴趣表征
            reg_A_emb: [N, hidden_dim] A域代表物品表征 (N通常为batch_size)
            reg_B_emb: [M, hidden_dim] B域代表物品表征 (M通常为batch_size)
        """
        # 检查输入有效性 (N, M >= 1)
        if reg_A_emb.size(0) < 1 or reg_B_emb.size(0) < 1:
            return torch.tensor(0.0, device=user_interest.device)
            
        # 标准化表征
        user_interest = F.normalize(user_interest, dim=1)
        reg_A_emb = F.normalize(reg_A_emb, dim=1)
        reg_B_emb = F.normalize(reg_B_emb, dim=1)
        
        # 计算用户兴趣与各域物品的相似度 (恢复 Matmul)
        # Note: If inputs are [bs, dim], this calculates cross-batch similarity!
        sim_U_A = torch.matmul(user_interest, reg_A_emb.t())  # Potentially [bs, N]
        sim_U_B = torch.matmul(user_interest, reg_B_emb.t())  # Potentially [bs, M]
        
        # 基础损失：拉近用户兴趣与各域代表物品的距离
        # 使用InfoNCE对比损失形式
        pos_A_exp = torch.exp(sim_U_A / self.tau)
        pos_B_exp = torch.exp(sim_U_B / self.tau)
        
        combined_sim = torch.cat([sim_U_A, sim_U_B], dim=1)  
        combined_exp = torch.exp(combined_sim / self.tau)
        combined_exp_sum = combined_exp.sum(dim=1).clamp(min=1e-9) # [bs]

        # Sticking to the likely *original code's* behavior for now, which might be averaging over rows:
        # This part might need revision if only diagonal of sim_U_A/B are true positives for InfoNCE
        loss_A_info_nce = -torch.log(pos_A_exp.sum(dim=1) / combined_exp_sum).mean()
        loss_B_info_nce = -torch.log(pos_B_exp.sum(dim=1) / combined_exp_sum).mean()
        
        base_loss = (loss_A_info_nce + loss_B_info_nce) / 2.0
        total_loss = base_loss
        
        # --- Hard Negative Mining Section ---
        bs = user_interest.size(0)
        N = reg_A_emb.size(0)
        M = reg_B_emb.size(0)

        # Conditions for meaningful HNM (batch size > 1 and items correspond to users for off-diagonal negatives)
        can_hnm_A = (self.hard_negative_weight > 0 and bs > 1 and N == bs)
        can_hnm_B = (self.hard_negative_weight > 0 and bs > 1 and M == bs)

        if can_hnm_A or can_hnm_B:
            h_loss_A = torch.tensor(0.0, device=base_loss.device)
            if can_hnm_A:
                h_loss_A = self._calculate_hnm_loss(sim_U_A, self.margin)
            
            h_loss_B = torch.tensor(0.0, device=base_loss.device)
            if can_hnm_B:
                h_loss_B = self._calculate_hnm_loss(sim_U_B, self.margin)

            # Combine hard negative losses, making B's contribution stronger
            # Example: weight A's hard loss by 0.5, B's by 1.0
            coeff_A = 0.5
            coeff_B = 1.0
            weighted_hard_loss = (coeff_A * h_loss_A + coeff_B * h_loss_B)
            
            if weighted_hard_loss.abs() > 1e-9 : # Only add if hard loss is significant
                total_loss = total_loss + self.hard_negative_weight * weighted_hard_loss
        
        return total_loss

class CrossDomainAlignmentLoss(nn.Module):
    """
    结合MoCo风格机制和分布一致性的跨域物品对齐损失
    """
    def __init__(self, dim=128, K=4096, m=0.999, T=0.07, dist_weight=0.2):
        """
        Args:
            dim: 特征维度
            K: 队列大小
            m: 动量更新参数
            T: 温度系数
            dist_weight: 分布一致性损失权重
        """
        super().__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.dist_weight = dist_weight
        
        # 初始化队列
        self.register_buffer("queue_A", torch.randn(dim, K))
        self.register_buffer("queue_B", torch.randn(dim, K))
        self.queue_A = F.normalize(self.queue_A, dim=0)
        self.queue_B = F.normalize(self.queue_B, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # 动量编码器 (使用浅拷贝projector作为初始值)
        self.momentum_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # 复制参数
        for param_q, param_k in zip(self.projector.parameters(), 
                                   self.momentum_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        动量更新编码器参数
        """
        for param_q, param_k in zip(self.projector.parameters(), 
                                   self.momentum_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_A, keys_B):
        """
        更新队列
        """
        batch_size = keys_A.shape[0]
        
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            # 处理队列循环
            remainder = ptr + batch_size - self.K
            self.queue_A[:, ptr:] = keys_A[:batch_size-remainder].T
            self.queue_A[:, :remainder] = keys_A[batch_size-remainder:].T
            
            self.queue_B[:, ptr:] = keys_B[:batch_size-remainder].T
            self.queue_B[:, :remainder] = keys_B[batch_size-remainder:].T
            
            ptr = remainder
        else:
            # 正常入队
            self.queue_A[:, ptr:ptr + batch_size] = keys_A.T
            self.queue_B[:, ptr:ptr + batch_size] = keys_B.T
            ptr = (ptr + batch_size) % self.K
            
        self.queue_ptr[0] = ptr
    
    def coral_loss(self, source, target):
        """
        计算CORAL损失 (CORrelation ALignment Loss)
        用于对齐两个域的二阶统计特性（协方差）
        
        Args:
            source: 源域特征 [N, D]
            target: 目标域特征 [M, D]
        """
        d = source.shape[1]
        
        # 计算协方差矩阵
        source = source - torch.mean(source, dim=0)
        source_cov = torch.mm(source.t(), source) / (source.shape[0] - 1)
        
        target = target - torch.mean(target, dim=0)
        target_cov = torch.mm(target.t(), target) / (target.shape[0] - 1)
        
        # 计算Frobenius范数
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * d * d)
        
        return loss
    
    def forward(self, embeddings_A, embeddings_B):
        """
        计算跨域对齐损失
        
        Args:
            embeddings_A: A域物品嵌入 [N, D]
            embeddings_B: B域物品嵌入 [M, D]
        """
        # 如果输入样本太少，返回零损失
        if embeddings_A.size(0) < 2 or embeddings_B.size(0) < 2:
            return torch.tensor(0.0, device=embeddings_A.device)
        
        # 特征归一化
        q_A = F.normalize(self.projector(embeddings_A), dim=1)
        q_B = F.normalize(self.projector(embeddings_B), dim=1)
        
        # 不计算梯度的前向传播
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # 计算动量嵌入
            k_A = F.normalize(self.momentum_projector(embeddings_A), dim=1)
            k_B = F.normalize(self.momentum_projector(embeddings_B), dim=1)
        
        # 对比损失 - InfoNCE公式
        # 正样本：跨域对应的物品
        # 负样本：队列中的所有样本
        
        # A->B方向的对比损失
        l_pos_A = torch.einsum('nc,nc->n', [q_A, k_B]).unsqueeze(-1)
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue_B])
        logits_A = torch.cat([l_pos_A, l_neg_A], dim=1)
        logits_A /= self.T
        
        # B->A方向的对比损失
        l_pos_B = torch.einsum('nc,nc->n', [q_B, k_A]).unsqueeze(-1)
        l_neg_B = torch.einsum('nc,ck->nk', [q_B, self.queue_A])
        logits_B = torch.cat([l_pos_B, l_neg_B], dim=1)
        logits_B /= self.T
        
        # 使用零作为标签，表示第一个是正样本
        labels = torch.zeros(logits_A.shape[0], dtype=torch.long, device=embeddings_A.device)
        
        # 计算交叉熵损失
        loss_A = F.cross_entropy(logits_A, labels)
        loss_B = F.cross_entropy(logits_B, labels)
        
        # 对比损失
        contrastive_loss = (loss_A + loss_B) / 2.0
        
        # 分布一致性损失 - 使用CORAL
        dist_loss = self.coral_loss(q_A, q_B)
        
        # 总损失
        total_loss = contrastive_loss + self.dist_weight * dist_loss
        
        # 更新队列
        self._dequeue_and_enqueue(k_A, k_B)
        
        return total_loss

class Enhanced_CrossDomain_Alignment(nn.Module):
    """
    增强版跨域对齐模块，结合SimCLR风格的对比学习和分布级对齐
    适用于较小批次的训练情况，不依赖大型队列
    """
    def __init__(self, dim=128, temperature=0.1, dist_weight=0.3):
        """
        Args:
            dim: 特征维度
            temperature: 温度参数
            dist_weight: 分布一致性损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.dist_weight = dist_weight
        
        # 非线性投影头
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # 特征变换增强器 - 用于创建不同的视角
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )
    
    def mmd_loss(self, x, y):
        """
        计算最大平均差异(MMD)损失
        用于匹配两个分布的一阶和二阶矩
        
        Args:
            x: 第一组特征 [n, d]
            y: 第二组特征 [m, d]
        """
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * xy
        
        XX = torch.zeros(xx.shape).to(x.device)
        YY = torch.zeros(xx.shape).to(x.device)
        XY = torch.zeros(xy.shape).to(x.device)
        
        bandwidth_range = [0.01, 0.1, 1, 10]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
        
        return torch.mean(XX + YY - 2. * XY)
    
    def forward(self, embeddings_A, embeddings_B):
        """
        计算增强的跨域对齐损失
        
        Args:
            embeddings_A: A域物品嵌入 [batch_size, dim]
            embeddings_B: B域物品嵌入 [batch_size, dim]
        """
        if embeddings_A.size(0) < 2 or embeddings_B.size(0) < 2:
            return torch.tensor(0.0, device=embeddings_A.device)
        
        # 投影和归一化
        z_A = F.normalize(self.projector(embeddings_A), dim=1)
        z_B = F.normalize(self.projector(embeddings_B), dim=1)
        
        # 创建增强视角
        z_A_aug = F.normalize(self.transform(embeddings_A), dim=1)
        z_B_aug = F.normalize(self.transform(embeddings_B), dim=1)
        
        # 合并所有表征用于对比学习
        batch_size = embeddings_A.size(0)
        features = torch.cat([z_A, z_B, z_A_aug, z_B_aug], dim=0)
        
        # 计算相似度矩阵
        sim_matrix = torch.exp(torch.mm(features, features.t()) / self.temperature)
        
        # 创建掩码排除自身相似度
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        mask = 1 - mask
        
        # 应用掩码
        sim_matrix = sim_matrix * mask
        
        # 定义正样本对
        # 原始A与增强A是正样本对，原始B与增强B是正样本对，A与B也是正样本对
        pos_mask = torch.zeros_like(sim_matrix)
        
        # A与A增强视角
        pos_mask[:batch_size, 2*batch_size:3*batch_size] = torch.eye(batch_size)
        # B与B增强视角
        pos_mask[batch_size:2*batch_size, 3*batch_size:] = torch.eye(batch_size)
        # A与B跨域对应
        pos_mask[:batch_size, batch_size:2*batch_size] = torch.eye(batch_size)
        # B与A跨域对应
        pos_mask[batch_size:2*batch_size, :batch_size] = torch.eye(batch_size)
        # A增强与B增强跨域对应
        pos_mask[2*batch_size:3*batch_size, 3*batch_size:] = torch.eye(batch_size)
        # B增强与A增强跨域对应
        pos_mask[3*batch_size:, 2*batch_size:3*batch_size] = torch.eye(batch_size)
        
        # 计算正样本相似度总和
        pos_sim = torch.sum(sim_matrix * pos_mask, dim=1)
        
        # 计算所有相似度总和
        total_sim = torch.sum(sim_matrix, dim=1)
        
        # 计算InfoNCE损失
        loss = -torch.log(pos_sim / total_sim).mean()
        
        # 分布对齐损失
        dist_loss = self.mmd_loss(z_A, z_B) + self.mmd_loss(z_A_aug, z_B_aug)
        
        # 总损失
        total_loss = loss + self.dist_weight * dist_loss
        
        return total_loss

class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, torch.finfo(attention.dtype).min)
        attention = torch.softmax(attention / torch.sqrt(torch.tensor(Q.size(-1), dtype=attention.dtype, device=attention.device)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        # Corrected mask generation: should be [B, 1, 1, S_kv] to broadcast correctly
        # S_kv is y.size(1) or log_seqs.size(1)
        attention_mask = (log_seqs == 0).unsqueeze(1).unsqueeze(2)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output
