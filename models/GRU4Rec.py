# here put the import lib
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_


class SASRecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        
        super().__init__()

        self.dev = device
        
        # 保留原始args参数的使用，适配到GRU4Rec
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.num_layers = getattr(args, 'trm_num', 2)  # 使用trm_num作为GRU层数
        
        # 假设vocab_size通过args传入，如果没有则需要在调用时提供
        self.vocab_size = getattr(args, 'vocab_size', 128)
        if self.vocab_size is None:
            raise ValueError("vocab_size must be provided in args for GRU4Rec")
        
        # GRU4Rec核心组件
        self.item_embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
        
        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_hh' in name:
                    xavier_uniform_(param)
                elif 'weight_ih' in name:
                    xavier_uniform_(param)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, seqs, log_seqs, timeline_mask=None):
        """
        Args:
            seqs: 项目序列，如果是索引则进行embedding，如果已经是embedding则直接使用
            log_seqs: 用于生成mask的序列（通常与seqs相同但可能是原始索引）
            timeline_mask: 可选的时间轴mask
        Returns:
            log_feats: GRU的输出特征
        """
        
        # 生成timeline_mask
        if timeline_mask is None:
            timeline_mask = (log_seqs == 0)
        
        # 如果seqs是索引，进行embedding；如果已经是embedding向量，直接使用
        if seqs.dtype == torch.long:
            # seqs是索引，需要embedding
            item_seq_emb = self.item_embedding(seqs)
        else:
            # seqs已经是embedding向量
            item_seq_emb = seqs
            
        # 应用mask：将padding位置的embedding置零
        item_seq_emb = item_seq_emb * ~timeline_mask.unsqueeze(-1)
        
        # Dropout
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        
        # GRU前向传播
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        
        # 通过dense层
        gru_output = self.dense(gru_output)
        
        # 应用mask确保padding位置输出为0
        gru_output = gru_output * ~timeline_mask.unsqueeze(-1)
        
        # 最后的layer normalization
        log_feats = self.last_layernorm(gru_output)
        
        return log_feats

    def get_sequence_output(self, seqs, log_seqs, seq_lens):
        """
        获取序列的最后一个有效位置的输出（用于预测）
        Args:
            seqs: 输入序列
            log_seqs: 用于mask的序列
            seq_lens: 序列长度
        Returns:
            最后有效位置的输出
        """
        log_feats = self.forward(seqs, log_seqs)
        
        # 获取每个序列最后一个有效位置的输出
        batch_size = log_feats.size(0)
        seq_output = log_feats[torch.arange(batch_size), seq_lens - 1]
        
        return seq_output

    

    