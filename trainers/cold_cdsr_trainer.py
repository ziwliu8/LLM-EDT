# here put the import lib
import os
import pickle
import torch
from tqdm import tqdm
import time # Add time import
import numpy as np
from trainers.sequence_trainer import SeqTrainer
from models.LLMCDSR import LLM4CDSR
from models.one4all import One4All
from utils.utils import record_csv, metric_report, metric_domain_report, metric_len_5group, metric_pop_5group
from models.domain_specific_adapter import DomainSpecificAdapter

class CDSRTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)


    def _create_model(self):
        """Create the model according to the model_name"""
        print(f"Model name: {self.args.model_name}")  # 调试信息
        self.item_num_dict = self.generator.get_item_num_dict()

        if self.args.model_name == "llm4cdsr":
            from models.LLMCDSR import LLM4CDSR
            self.model = LLM4CDSR(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "One4All":
            from models.one4all import One4All
            self.model = One4All(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "C2DSR":
            from models.C2DSR_adapted import C2DSR_Adapted
            self.model = C2DSR_Adapted(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "One4All_Finetuner":
            from models.domain_enhance import One4All_Finetuner
            self.model = One4All_Finetuner(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "One4AllAttentionOnly":
            from models.one4all_attention_only import One4AllAttentionOnly
            self.model = One4AllAttentionOnly(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "One4AllEmbeddingOnly":
            from models.one4all_embedding_only import One4AllEmbeddingOnly
            self.model = One4AllEmbeddingOnly(self.user_num, self.item_num_dict, self.device, self.args)
        elif self.args.model_name == "DomainSpecificAdapter":
            # Ensure necessary fields are in args (although usually handled by DomainAdapterTrainer)
            if not hasattr(self.args, 'user_num'): self.args.user_num = self.user_num
            if not hasattr(self.args, 'item_numA'): self.args.item_numA = self.item_num_dict["0"]
            if not hasattr(self.args, 'item_numB'): self.args.item_numB = self.item_num_dict["1"]
            if not hasattr(self.args, 'gpu_id'): # Model uses gpu_id to set device
                if self.device.type == 'cuda': self.args.gpu_id = self.device.index if self.device.index is not None else 0
                else: self.args.gpu_id = 0 # Default for CPU
            
            from models.domain_specific_adapter import DomainSpecificAdapter
            # Pass only args, as expected by the constructor
            self.model = DomainSpecificAdapter(args=self.args)
        else:
            raise ValueError(f"Invalid model_name: {self.args.model_name}")
        
        self.model.to(self.device)

        if self.device.type == 'cuda':
            # Ensure the device object passed is correctly interpreted by CUDA API
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}')
            self.logger.info("--- GPU Memory After Model Creation and to(device) (CDSRTrainer) ---")
            allocated_mem_gb = torch.cuda.memory_allocated(device_for_stats) / (1024**3)
            # Reset peak stats an instant before measuring to capture the model load itself if possible,
            # though model creation might have already allocated.
            # For a cleaner measure of model static size, reset before model instantiation and .to(device)
            # However, here we measure *after* .to(device), so peak might include some prior operations.
            # Let's get current peak as is, which reflects state after model is on device.
            peak_allocated_mem_gb = torch.cuda.max_memory_allocated(device_for_stats) / (1024**3)
            reserved_mem_gb = torch.cuda.memory_reserved(device_for_stats) / (1024**3)
            peak_reserved_mem_gb = torch.cuda.max_memory_reserved(device_for_stats) / (1024**3)
            self.logger.info(
                f"Model Load (CDSRTrainer): 已分配 {allocated_mem_gb:.3f} GB (当前峰值 {peak_allocated_mem_gb:.3f} GB), "
                f"已预留 {reserved_mem_gb:.3f} GB (当前峰值 {peak_reserved_mem_gb:.3f} GB)"
            )
            self.logger.info("---------------------------------------------------------------------")

    

    def eval(self, epoch=0, test=False):
        eval_start_time = time.time() # Start eval timer
        total_batch_eval_time_ms = 0.0

        if self.device.type == 'cuda':
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}')
            torch.cuda.reset_peak_memory_stats(device_for_stats) # Reset for eval phase peak

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        # 【修改2】获取item流行度字典（在循环开始前）
        # 假设generator有get_item_pop_dict方法，或者从数据集中获取
        # item_pop_dict = self.generator.get_item_pop_dict()  # 需要在generator中实现这个方法
        # 或者从其他地方加载预计算的流行度
        # 如果没有现成的，可以暂时创建一个dummy dict用于测试
        item_pop_dict = {i: np.random.randint(1, 200) for i in range(1, self.item_num+1)}

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            
            with torch.no_grad():
                #print(inputs["pos"].shape, inputs["neg"].shape)
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indicesA"] = torch.cat([inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)
                inputs["item_indicesB"] = torch.cat([inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)
                
                batch_eval_inference_start_time = time.time()
                pred_logits = -self.model.predict(**inputs)
                batch_eval_inference_end_time = time.time()
                total_batch_eval_time_ms += (batch_eval_inference_end_time - batch_eval_inference_start_time) * 1000

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        eval_end_time = time.time()
        eval_duration_ms = (eval_end_time - eval_start_time) * 1000
        # Note: len(test_loader) can be 0 if the loader is empty, handle division by zero.
        avg_batch_eval_time_ms = (total_batch_eval_time_ms / len(test_loader)) if len(test_loader) > 0 else 0

        eval_phase_name = "测试" if test else f"Epoch {epoch} 评估"
        self.logger.info(f"{eval_phase_name} (CDSRTrainer) 总耗时: {eval_duration_ms:.2f} ms")
        self.logger.info(f"{eval_phase_name} (CDSRTrainer) 平均批次推理耗时: {avg_batch_eval_time_ms:.2f} ms/batch")
        
        if self.device.type == 'cuda':
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}')
            allocated_mem_gb = torch.cuda.memory_allocated(device_for_stats) / (1024**3)
            peak_allocated_mem_gb = torch.cuda.max_memory_allocated(device_for_stats) / (1024**3)
            reserved_mem_gb = torch.cuda.memory_reserved(device_for_stats) / (1024**3)
            peak_reserved_mem_gb = torch.cuda.max_memory_reserved(device_for_stats) / (1024**3)
            self.logger.info(
                f"{eval_phase_name} (CDSRTrainer), GPU显存: 当前已分配 {allocated_mem_gb:.3f} GB (本阶段峰值 {peak_allocated_mem_gb:.3f} GB), "
                f"当前已预留 {reserved_mem_gb:.3f} GB (本阶段峰值 {peak_reserved_mem_gb:.3f} GB)"
            )

        self.logger.info('')
        
        # 【修改3】替换原来的metric_report调用
        # 原来的代码：
        # res_dict = metric_report(pred_rank.detach().cpu().numpy())
        
        # 【修改4】新的5组评估代码（替换上面的res_dict = metric_report行）：
        pred_rank_np = pred_rank.detach().cpu().numpy()
        seq_len_np = seq_len.detach().cpu().numpy()
        target_items_np = target_items.detach().cpu().numpy()
        
        # 按序列长度分组评估
        HR_len, NDCG_len, count_len = metric_len_5group(pred_rank_np, seq_len_np, 
                                                        thresholds=[5, 10, 15, 20], topk=10)
        
        # 按物品流行度分组评估
        HR_pop, NDCG_pop, count_pop = metric_pop_5group(pred_rank_np, item_pop_dict, target_items_np,
                                                        thresholds=[5, 10, 20], topk=10)
        
        # 构建详细的res_dict
        res_dict = {}
        # 添加原始的整体指标
        overall_metrics = metric_report(pred_rank_np)
        res_dict.update(overall_metrics)
        
        # 添加按序列长度分组的指标
        len_groups = ['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
        for i, group in enumerate(len_groups):
            res_dict[f'HR@10_len_{group}'] = HR_len[i]
            res_dict[f'NDCG@10_len_{group}'] = NDCG_len[i]
            res_dict[f'Count_len_{group}'] = count_len[i]
        
        # 添加按流行度分组的指标
        pop_groups = ['Tail', 'Unpopular', 'Popular', 'Head']
        for i, group in enumerate(pop_groups):
            res_dict[f'HR@10_pop_{group}'] = HR_pop[i]
            res_dict[f'NDCG@10_pop_{group}'] = NDCG_pop[i]
            res_dict[f'Count_pop_{group}'] = count_pop[i]

        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        # distinguish the domain A and B
        pred_rank_A = pred_rank[target_domain==0]
        pred_rank_B = pred_rank[target_domain==1]
        
        # 【修改5】为域A和域B也应用5组评估
        # 原来的代码：
        # res_dict_A = metric_domain_report(pred_rank_A.detach().cpu().numpy(), domain="A")
        # res_dict_B = metric_domain_report(pred_rank_B.detach().cpu().numpy(), domain="B")
        
        # 【修改6】新的域A、B的5组评估代码（替换上面两行）：
        # 域A的详细评估
        domain_A_mask = target_domain == 0
        pred_rank_A_np = pred_rank_A.detach().cpu().numpy()
        seq_len_A_np = seq_len[domain_A_mask].detach().cpu().numpy()
        target_items_A_np = target_items[domain_A_mask].detach().cpu().numpy()
        
        HR_len_A, NDCG_len_A, count_len_A = metric_len_5group(pred_rank_A_np, seq_len_A_np, 
                                                              thresholds=[5, 10, 15, 20], topk=10)
        HR_pop_A, NDCG_pop_A, count_pop_A = metric_pop_5group(pred_rank_A_np, item_pop_dict, target_items_A_np,
                                                              thresholds=[5, 10, 20], topk=10)
        
        res_dict_A = metric_domain_report(pred_rank_A_np, domain="A")
        for i, group in enumerate(len_groups):
            res_dict_A[f'HR@10_len_{group}_A'] = HR_len_A[i]
            res_dict_A[f'NDCG@10_len_{group}_A'] = NDCG_len_A[i]
            res_dict_A[f'Count_len_{group}_A'] = count_len_A[i]
        for i, group in enumerate(pop_groups):
            res_dict_A[f'HR@10_pop_{group}_A'] = HR_pop_A[i]
            res_dict_A[f'NDCG@10_pop_{group}_A'] = NDCG_pop_A[i]
            res_dict_A[f'Count_pop_{group}_A'] = count_pop_A[i]
        
        # 域B的详细评估
        domain_B_mask = target_domain == 1
        pred_rank_B_np = pred_rank_B.detach().cpu().numpy()
        seq_len_B_np = seq_len[domain_B_mask].detach().cpu().numpy()
        target_items_B_np = target_items[domain_B_mask].detach().cpu().numpy()
        
        HR_len_B, NDCG_len_B, count_len_B = metric_len_5group(pred_rank_B_np, seq_len_B_np, 
                                                              thresholds=[5, 10, 15, 20], topk=10)
        HR_pop_B, NDCG_pop_B, count_pop_B = metric_pop_5group(pred_rank_B_np, item_pop_dict, target_items_B_np,
                                                              thresholds=[5, 10, 20], topk=10)
        
        res_dict_B = metric_domain_report(pred_rank_B_np, domain="B")
        for i, group in enumerate(len_groups):
            res_dict_B[f'HR@10_len_{group}_B'] = HR_len_B[i]
            res_dict_B[f'NDCG@10_len_{group}_B'] = NDCG_len_B[i]
            res_dict_B[f'Count_len_{group}_B'] = count_len_B[i]
        for i, group in enumerate(pop_groups):
            res_dict_B[f'HR@10_pop_{group}_B'] = HR_pop_B[i]
            res_dict_B[f'NDCG@10_pop_{group}_B'] = NDCG_pop_B[i]
            res_dict_B[f'Count_pop_{group}_B'] = count_pop_B[i]


        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("Domain A Performance:")
            for k, v in res_dict_A.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Domain B Performance:")
            for k, v in res_dict_B.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_dict_A, **res_dict_B}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
    

    def save_item_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        all_index = torch.arange(start=1, end=self.item_num+1).to(self.device)
        item_emb = self.model._get_embedding(all_index, self.args.domain)
        item_emb = item_emb.detach().cpu().numpy()
        pickle.dump(item_emb, open("./data/{}/handled/itm_emb_{}.pkl".format(self.args.dataset, self.args.model_name), "wb"))


    def eval_cold(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running cold test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.generator.make_coldloader()
    
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indicesA"] = torch.cat([inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)
                inputs["item_indicesB"] = torch.cat([inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        
        # 【修改7】冷启动评估也使用5组评估
        # 先获取item流行度字典
        # item_pop_dict = self.generator.get_item_pop_dict()  # 需要在generator中实现
        # 或者使用dummy数据：
        item_pop_dict = {i: np.random.randint(1, 200) for i in range(1, self.item_num+1)}
        
        # 原来的代码：
        # res_dict = metric_report(pred_rank.detach().cpu().numpy())
        
        # 【修改8】新的冷启动5组评估代码（替换上面的res_dict = metric_report行）：
        pred_rank_np = pred_rank.detach().cpu().numpy()
        seq_len_np = seq_len.detach().cpu().numpy()
        target_items_np = target_items.detach().cpu().numpy()
        
        # 按序列长度分组评估
        HR_len, NDCG_len, count_len = metric_len_5group(pred_rank_np, seq_len_np, 
                                                        thresholds=[5, 10, 15, 20], topk=10)
        
        # 按物品流行度分组评估
        HR_pop, NDCG_pop, count_pop = metric_pop_5group(pred_rank_np, item_pop_dict, target_items_np,
                                                        thresholds=[5, 10, 20], topk=10)
        
        # 构建详细的res_dict
        res_dict = {}
        # 添加原始的整体指标
        overall_metrics = metric_report(pred_rank_np)
        res_dict.update(overall_metrics)
        
        # 添加按序列长度分组的指标
        len_groups = ['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
        for i, group in enumerate(len_groups):
            res_dict[f'cold_HR@10_len_{group}'] = HR_len[i]
            res_dict[f'cold_NDCG@10_len_{group}'] = NDCG_len[i]
            res_dict[f'cold_Count_len_{group}'] = count_len[i]
        
        # 添加按流行度分组的指标
        pop_groups = ['Tail', 'Unpopular', 'Popular', 'Head']
        for i, group in enumerate(pop_groups):
            res_dict[f'cold_HR@10_pop_{group}'] = HR_pop[i]
            res_dict[f'cold_NDCG@10_pop_{group}'] = NDCG_pop[i]
            res_dict[f'cold_Count_pop_{group}'] = count_pop[i]

        # distinguish the domain A and B
        pred_rank_A = pred_rank[target_domain==0]
        pred_rank_B = pred_rank[target_domain==1]
        
        # 【修改9】冷启动的域A、B也使用5组评估
        # 原来的代码：
        # res_dict_A = metric_domain_report(pred_rank_A.detach().cpu().numpy(), domain="A")
        # res_dict_B = metric_domain_report(pred_rank_B.detach().cpu().numpy(), domain="B")
        
        # 【修改10】新的冷启动域A、B的5组评估代码（替换上面两行）：
        # 域A的详细评估
        domain_A_mask = target_domain == 0
        pred_rank_A_np = pred_rank_A.detach().cpu().numpy()
        seq_len_A_np = seq_len[domain_A_mask].detach().cpu().numpy()
        target_items_A_np = target_items[domain_A_mask].detach().cpu().numpy()
        
        HR_len_A, NDCG_len_A, count_len_A = metric_len_5group(pred_rank_A_np, seq_len_A_np, 
                                                              thresholds=[5, 10, 15, 20], topk=10)
        HR_pop_A, NDCG_pop_A, count_pop_A = metric_pop_5group(pred_rank_A_np, item_pop_dict, target_items_A_np,
                                                              thresholds=[5, 10, 20], topk=10)
        
        res_dict_A = metric_domain_report(pred_rank_A_np, domain="A")
        for i, group in enumerate(len_groups):
            res_dict_A[f'cold_HR@10_len_{group}_A'] = HR_len_A[i]
            res_dict_A[f'cold_NDCG@10_len_{group}_A'] = NDCG_len_A[i]
            res_dict_A[f'cold_Count_len_{group}_A'] = count_len_A[i]
        for i, group in enumerate(pop_groups):
            res_dict_A[f'cold_HR@10_pop_{group}_A'] = HR_pop_A[i]
            res_dict_A[f'cold_NDCG@10_pop_{group}_A'] = NDCG_pop_A[i]
            res_dict_A[f'cold_Count_pop_{group}_A'] = count_pop_A[i]
        
        # 域B的详细评估
        domain_B_mask = target_domain == 1
        pred_rank_B_np = pred_rank_B.detach().cpu().numpy()
        seq_len_B_np = seq_len[domain_B_mask].detach().cpu().numpy()
        target_items_B_np = target_items[domain_B_mask].detach().cpu().numpy()
        
        HR_len_B, NDCG_len_B, count_len_B = metric_len_5group(pred_rank_B_np, seq_len_B_np, 
                                                              thresholds=[5, 10, 15, 20], topk=10)
        HR_pop_B, NDCG_pop_B, count_pop_B = metric_pop_5group(pred_rank_B_np, item_pop_dict, target_items_B_np,
                                                              thresholds=[5, 10, 20], topk=10)
        
        res_dict_B = metric_domain_report(pred_rank_B_np, domain="B")
        for i, group in enumerate(len_groups):
            res_dict_B[f'cold_HR@10_len_{group}_B'] = HR_len_B[i]
            res_dict_B[f'cold_NDCG@10_len_{group}_B'] = NDCG_len_B[i]
            res_dict_B[f'cold_Count_len_{group}_B'] = count_len_B[i]
        for i, group in enumerate(pop_groups):
            res_dict_B[f'cold_HR@10_pop_{group}_B'] = HR_pop_B[i]
            res_dict_B[f'cold_NDCG@10_pop_{group}_B'] = NDCG_pop_B[i]
            res_dict_B[f'cold_Count_pop_{group}_B'] = count_pop_B[i]


        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("Domain A Performance:")
        for k, v in res_dict_A.items():
            self.logger.info('\t %s: %.5f' % (k, v))
        self.logger.info("Domain B Performance:")
        for k, v in res_dict_B.items():
            self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_dict_A, **res_dict_B}

        # modify the key as cold
        key_list = list(res_dict.keys())
        for key in key_list:
            res_dict.update({"cold_{}".format(key): res_dict.pop(key)})

        record_csv(self.args, res_dict)
        
        return res_dict