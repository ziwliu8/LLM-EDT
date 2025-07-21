#基准
# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2


class One4All_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
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
    
class One4All(One4All_base):

    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)

        self.beta = args.beta
        #llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        #self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        #self.user_emb_llm.weight.requires_grad = False

        #self.user_adapter = nn.Sequential(
        #    nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
        #    nn.Linear(int(llm_user_emb.shape[1] / 2), args.hidden_size)
        #)

        #self.user_loss_func = Contrastive_Loss2(tau=args.tau)

        
        #self.filter_init_modules.append("user_emb_llm")
        self._init_weights()

    def forward(self, seq, pos, neg, positions, reg_A, reg_B, user_id, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # LLM user embedding guidance
        #log_feats = self.log2feats(seq, positions)
        #final_feat = log_feats[:, -1, :]
        #llm_feats = self.user_emb_llm(user_id)
        #llm_feats = self.user_adapter(llm_feats)
        #user_loss = self.user_loss_func(llm_feats, final_feat)

        #loss += self.beta * user_loss

        return loss



