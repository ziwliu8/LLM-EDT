# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # 线性变换并分头
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力权重
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        output = self.output_linear(context)
        return output


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, hidden_size, inner_size, dropout_rate):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Transformer层"""
    def __init__(self, hidden_size, num_heads, inner_size, dropout_rate, layer_norm_eps=1e-12):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = FeedForward(hidden_size, inner_size, dropout_rate)
        self.attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states, attention_mask=None):
        # 多头注意力 + 残差连接
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layernorm(hidden_states + attention_output)
        
        # 前馈网络 + 残差连接
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.output_layernorm(attention_output + feed_forward_output)
        
        return output


class SASRecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        
        super().__init__()

        self.dev = device
        
        # 保留原始args参数的使用，适配到BERT4Rec
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.n_layers = getattr(args, 'trm_num', 2)  # 使用trm_num作为Transformer层数
        self.n_heads = getattr(args, 'num_heads', 8)  # 多头注意力头数
        self.inner_size = getattr(args, 'inner_size', self.hidden_size * 4)  # FFN内部维度
        self.layer_norm_eps = getattr(args, 'layer_norm_eps', 1e-12)
        self.initializer_range = getattr(args, 'initializer_range', 0.02)
        
        # 词汇表大小和最大序列长度
        self.vocab_size = getattr(args, 'vocab_size', 128)
        self.max_seq_length = getattr(args, 'max_seq_length', 50)
        
        # BERT4Rec核心组件
        # mask token是vocab_size，所以embedding需要+1
        self.item_embedding = nn.Embedding(
            self.vocab_size + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.n_heads,
                inner_size=self.inner_size,
                dropout_rate=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps
            ) for _ in range(self.n_layers)
        ])
        
        # Layer normalization和dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 输出层
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # mask token
        self.mask_token = self.vocab_size
        
        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_attention_mask(self, item_seq, bidirectional=True):
        """生成注意力mask"""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        if not bidirectional:
            # 单向注意力（因果mask）
            seq_length = item_seq.size(1)
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), 
                                              device=item_seq.device)).unsqueeze(0).unsqueeze(0)
            extended_attention_mask = extended_attention_mask * causal_mask
            
        # 转换为注意力分数的mask（0表示可以注意，-10000表示不能注意）
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, seqs, log_seqs=None, timeline_mask=None, bidirectional=True):
        """
        Args:
            seqs: 项目序列（可以是索引或已经是embedding向量）
            log_seqs: 用于生成mask的序列（保持兼容性）
            timeline_mask: 可选的时间轴mask
            bidirectional: 是否使用双向注意力
        Returns:
            log_feats: Transformer的输出特征
        """
        
        # 如果没有提供log_seqs，使用seqs
        if log_seqs is None:
            log_seqs = seqs
            
        # 判断seqs是索引还是已经是embedding向量
        if seqs.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            # seqs是索引，需要进行embedding和位置编码
            position_ids = torch.arange(seqs.size(1), dtype=torch.long, device=seqs.device)
            position_ids = position_ids.unsqueeze(0).expand_as(seqs)
            position_embedding = self.position_embedding(position_ids)
            
            # 项目嵌入
            item_emb = self.item_embedding(seqs)
            
            # 输入嵌入 = 项目嵌入 + 位置嵌入
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)
            
            # 使用seqs生成attention mask
            extended_attention_mask = self.get_attention_mask(seqs, bidirectional=bidirectional)
            
        else:
            # seqs已经是embedding向量，直接使用
            input_emb = seqs
            # 生成attention mask需要使用log_seqs
            extended_attention_mask = self.get_attention_mask(log_seqs, bidirectional=bidirectional)
        
        # 通过Transformer层
        hidden_states = input_emb
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(hidden_states, extended_attention_mask)
        
        # 输出处理
        ffn_output = self.output_ffn(hidden_states)
        ffn_output = F.gelu(ffn_output)
        log_feats = self.output_ln(ffn_output)
        
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
        log_feats = self.forward(seqs, log_seqs, bidirectional=True)
        
        # 获取每个序列最后一个有效位置的输出
        batch_size = log_feats.size(0)
        seq_output = log_feats[torch.arange(batch_size), seq_lens - 1]
        
        return seq_output
    
    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        在最后位置添加mask token（用于测试）
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    

    