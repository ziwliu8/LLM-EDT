#微调 （SeqB迁移）

# LLM4CDSR/models/domain_specific_adapter.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
from models.one4all import One4All
from models.utils import Contrastive_Loss2


# Simplified Adapter module
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_size, hidden_size)
        )
    
    def forward(self, x):
        return x + self.layer(x) # Residual connection


# New DomainSpecificAdapter implementation based on user description
class DomainSpecificAdapter(nn.Module):
    """
    Implements the new fine-tuning framework:
    1. Backbone produces initial representations.
    2. Domain Adapters refine seqA/seqB representations.
    3. Final prediction based on fused domain-specific features.
    4. Contrastive learning between sequence-based user representations and LLM user embeddings.
    """
    def __init__(self, args, pretrained_path=None):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.hidden_size = args.hidden_size
        self.adapter_bottleneck_size = getattr(args, 'adapter_size', 128)
        self.dropout_rate = getattr(args, 'dropout_rate', 0.1)

        # Initialize filter_init_modules attribute
        self.filter_init_modules = []

        print("--- STRATEGY: Domain Adapters with LLM User Contrastive Learning ---")

        # Load Pre-trained One4All Model
        self._load_pretrained_model(pretrained_path)
        
        # Print Pre-trained Backbone Parameters
        if self.base_model:
            backbone_total_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"Parameters in pre-trained backbone (One4All): {backbone_total_params}")
        else:
            print("Warning: self.base_model is not loaded, cannot count backbone parameters.")

        # Freeze Backbone
        self._freeze_backbone()

        # Define New Modules
        # 1. Domain Adapters
        self.adapter_A = Adapter(self.hidden_size, self.adapter_bottleneck_size)
        self.adapter_B = Adapter(self.hidden_size, self.adapter_bottleneck_size)
        
        # 2. Domain-specific positional embeddings
        self.pos_embA = nn.Embedding(args.max_len + 1, self.hidden_size)
        self.pos_embB = nn.Embedding(args.max_len + 1, self.hidden_size)
        
        # 3. User representation projection for contrastive learning
        self.user_seq_projector = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 4. LLM User Embedding and Adapter
        llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        self.user_emb_llm.weight.requires_grad = False
        
        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
            nn.Linear(int(llm_user_emb.shape[1] / 2), self.hidden_size)
        )

        # 5. Contrastive Loss for User Representation Learning
        self.user_loss_func = Contrastive_Loss2(tau=args.tau)
        self.beta = args.beta  # Weight for contrastive loss

        # Calculate and Print Parameters of Newly Added Modules
        newly_added_modules = [
            self.adapter_A, self.adapter_B,
            self.pos_embA, self.pos_embB,
            self.user_seq_projector, self.user_adapter
        ]
        
        added_modules_total_params = sum(p.numel() for module in newly_added_modules for p in module.parameters())
        print(f"Parameters in newly added fine-tuning modules: {added_modules_total_params}")

        # Add user_emb_llm to filter_init_modules to prevent initialization
        self.filter_init_modules.append("user_emb_llm")
        
        # Initialize weights and move to device
        self._init_weights()
        self.to(self.device)
        self._print_trainable_parameters()

    def _init_weights(self):
        """Initializes weights of the newly added modules."""
        print("Initializing weights for newly added modules...")
        modules_to_init = [
            self.adapter_A, self.adapter_B, self.user_adapter,
            self.pos_embA, self.pos_embB, self.user_seq_projector
        ]
        for module in modules_to_init:
            module.apply(self._xavier_init_fn)

    def _xavier_init_fn(self, module):
        """Helper function to apply Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def _load_pretrained_model(self, pretrained_path):
        """Loads the original pre-trained One4All model."""
        self.user_num = self.args.user_num
        self.item_num_dict = {"0": self.args.item_numA, "1": self.args.item_numB}
        self.base_model = One4All(self.user_num, self.item_num_dict, self.device, self.args)
        
        if pretrained_path is None: 
            pretrained_path = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        
        print(f"Loading pre-trained One4All model backbone from: {pretrained_path}")
        
        if not os.path.exists(pretrained_path): 
            print("Warning: Pretrained model file not found.")
            self.base_model.to(self.device)
            return
        
        try:
            model_state_dict = torch.load(pretrained_path, map_location=self.device)
            correct_state_dict = {}
            loaded_dict = model_state_dict.get('state_dict', model_state_dict)
            
            for k, v in loaded_dict.items():
                correct_state_dict[k.replace("module.", "")] = v

            missing_keys, unexpected_keys = self.base_model.load_state_dict(correct_state_dict, strict=False)
            
            if missing_keys: 
                print(f"Warning: Missing keys when loading base model: {missing_keys}")
            if unexpected_keys: 
                print(f"Warning: Unexpected keys when loading base model: {unexpected_keys}")
                
            print("Pre-trained One4All backbone state_dict loaded.")
            
        except Exception as e: 
            print(f"Error loading state_dict: {e}")
            raise e
            
        self.base_model.to(self.device)
    
    def _freeze_backbone(self):
        """Freezes parameters of the loaded base_model."""
        print("Freezing pre-trained backbone parameters...")
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False

    def _get_backbone_output(self, seq, positions, domain=None):
        """Helper to pass sequence through the base model's embedding and backbone."""
        needs_grad = False
        
        with torch.set_grad_enabled(needs_grad):
            item_embs = self.base_model._get_embedding(seq)  # [B, S, H]
            backbone_input = item_embs * (self.args.hidden_size ** 0.5)
            
            # Use domain-specific positional embeddings if specified
            if domain == "A":
                seq_pos_emb = self.pos_embA(positions.long())  # [B, S, H]
            elif domain == "B":
                seq_pos_emb = self.pos_embB(positions.long())  # [B, S, H]
            else:
                seq_pos_emb = self.base_model.pos_emb(positions.long())  # [B, S, H]
            
            backbone_input = backbone_input + seq_pos_emb
            
            # Use dropout if base model has it
            if hasattr(self.base_model, 'emb_dropout'):
                backbone_input = self.base_model.emb_dropout(backbone_input)
            
            # Pass through the backbone
            log_feats = self.base_model.backbone(backbone_input, seq)  # [B, S, H]
            
        return log_feats

    def _get_domain_representations(self, seq, positions, seqA, positionsA, seqB, positionsB):
        """Get domain-specific representations with adapters."""
        # Adjust seqB for global item IDs
        seqB_input_adjusted = torch.where(seqB != 0, seqB + self.args.item_numA, seqB)
        
        # Get backbone outputs
        hidden_global = self._get_backbone_output(seq, positions)
        hidden_global_A = self._get_backbone_output(seqA, positionsA, domain="A")
        hidden_global_B = self._get_backbone_output(seqB_input_adjusted, positionsB, domain="B")
        
        # Apply domain adapters
        h_prime_A = self.adapter_A(hidden_global_A)  # [B, S, H]
        h_prime_B = self.adapter_B(hidden_global_B)  # [B, S, H]
        
        return hidden_global, h_prime_A, h_prime_B

    def _compute_contrastive_loss(self, user_representations, user_id):
        """Compute contrastive loss between sequence-based and LLM user representations."""
        # Get LLM user embeddings
        llm_user_feats = self.user_emb_llm(user_id)  # [B, llm_dim]
        llm_user_feats = self.user_adapter(llm_user_feats)  # [B, H]
        
        # Project sequence-based user representations
        seq_user_feats = self.user_seq_projector(user_representations)  # [B, H]
        
        # Compute contrastive loss
        contrastive_loss = self.user_loss_func(seq_user_feats, llm_user_feats)
        
        return contrastive_loss

    def _calculate_domain_loss(self, domain_representations, pos_items, neg_items, domain_mask, domain_type):
        """Calculate domain-specific auxiliary loss."""
        if domain_type == "A":
            valid_mask = (pos_items != 0)
            pos_global = pos_items
            neg_global = neg_items
        else:  # domain_type == "B"
            valid_mask = (pos_items != 0)
            pos_global = torch.where(pos_items != 0, pos_items + self.args.item_numA, pos_items)
            neg_global = torch.where(neg_items != 0, neg_items + self.args.item_numA, neg_items)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # Get embeddings and compute logits
        pos_embs = self.base_model._get_embedding(pos_global)
        neg_embs = self.base_model._get_embedding(neg_global)
        
        pos_logits = (domain_representations * pos_embs).sum(dim=-1)
        neg_logits = (domain_representations * neg_embs).sum(dim=-1)
        
        # Create labels
        pos_labels = torch.ones(pos_logits.shape, device=self.device)
        neg_labels = torch.zeros(neg_logits.shape, device=self.device)
        
        # Calculate loss
        if hasattr(self.base_model, 'loss_func'):
            try:
                pos_loss = self.base_model.loss_func(pos_logits[valid_mask], pos_labels[valid_mask])
                neg_loss = self.base_model.loss_func(neg_logits[valid_mask], neg_labels[valid_mask])
                return (pos_loss.mean() + neg_loss.mean()) / 2
            except Exception as e:
                print(f"Error calculating domain {domain_type} loss: {e}")
                return torch.tensor(0.0, device=self.device)
        else:
            print(f"Warning: base_model.loss_func not found for domain {domain_type}")
            return torch.tensor(0.0, device=self.device)

    def _print_trainable_parameters(self):
        """Prints the number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    def forward(self, seq, positions, pos, neg,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                domain_mask, **kwargs):
        
        user_id = kwargs.get('user_id', None)
        
        # Get domain-specific representations
        hidden_global, h_prime_A, h_prime_B = self._get_domain_representations(
            seq, positions, seqA, positionsA, seqB, positionsB
        )
        
        # Get user representation from global sequence (last time step)
        user_seq_representation = hidden_global[:, -1, :]  # [B, H]
        
        # Compute contrastive loss between sequence-based and LLM user representations
        contrastive_loss = self._compute_contrastive_loss(user_seq_representation, user_id)
        
        # Calculate main prediction loss
        pos_embs = self.base_model._get_embedding(pos)
        neg_embs = self.base_model._get_embedding(neg)
        pos_logits = (hidden_global * pos_embs).sum(dim=-1)
        neg_logits = (hidden_global * neg_embs).sum(dim=-1)
        
        # Calculate domain-specific auxiliary losses
        mask_A_valid = (seqA.sum(dim=1) != 0)
        mask_B_valid = (seqB.sum(dim=1) != 0)
        
        loss_domain_A = torch.tensor(0.0, device=self.device)
        loss_domain_B = torch.tensor(0.0, device=self.device)
        
        if mask_A_valid.any():
            loss_domain_A = self._calculate_domain_loss(h_prime_A, posA, negA, domain_mask, "A")
        
        if mask_B_valid.any():
            loss_domain_B = self._calculate_domain_loss(h_prime_B, posB, negB, domain_mask, "B")
        
        # Combine losses
        aux_loss_weight_A = getattr(self.args, 'aux_loss_A_weight', 1.0)
        aux_loss_weight_B = getattr(self.args, 'aux_loss_B_weight', 1.0)
        
        total_loss = (aux_loss_weight_A * loss_domain_A + 
                     aux_loss_weight_B * loss_domain_B + 
                     self.beta * contrastive_loss)
        
        return total_loss

    def predict(self, seq, item_indices, positions, target_domain,
                seqA, positionsA, item_indicesA,
                seqB, positionsB, item_indicesB,
                **kwargs):
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                # Get domain-specific representations
                hidden_global, h_prime_A, h_prime_B = self._get_domain_representations(
                    seq, positions, seqA, positionsA, seqB, positionsB
                )
                
                # Get final user representation
                final_user_feat = hidden_global[:, -1, :]  # [B, H]
                
                # Main score calculation
                item_embs = self.base_model._get_embedding(item_indices)
                logits = item_embs.matmul(final_user_feat.unsqueeze(-1)).squeeze(-1)
                
                # Add domain-specific scores
                if item_indicesA is not None:
                    item_embA = self.base_model._get_embedding(item_indicesA)
                    final_user_featA = h_prime_A[:, -1, :]
                    logits_A = item_embA.matmul(final_user_featA.unsqueeze(-1)).squeeze(-1)
                    logits[target_domain == 0] += logits_A[target_domain == 0]
                
                if item_indicesB is not None:
                    # Convert local B IDs to global IDs if needed
                    item_indicesB_global = torch.where(
                        (item_indicesB > 0) & (item_indicesB <= self.args.item_numB),
                        item_indicesB + self.args.item_numA,
                        item_indicesB
                    )
                    
                    item_embB = self.base_model._get_embedding(item_indicesB_global)
                    final_user_featB = h_prime_B[:, -1, :]
                    logits_B = item_embB.matmul(final_user_featB.unsqueeze(-1)).squeeze(-1)
                    logits[target_domain == 1] += logits_B[target_domain == 1]
                
        return logits








