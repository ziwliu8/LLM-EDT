# trainers/domain_adapter_trainer.py
import os
import torch
from tqdm import tqdm
import time # Add time import
from trainers.cdsr_trainer import CDSRTrainer
from models.domain_specific_adapter import DomainSpecificAdapter
from utils.utils import record_csv, metric_report, metric_domain_report
# 【修改1】添加新的评估函数导入
# from utils.utils import record_csv, metric_report, metric_domain_report, metric_len_5group, metric_pop_5group

class DomainAdapterTrainer(CDSRTrainer):
    """
    领域适配器训练器，专门用于训练DomainSpecificAdapter模型
    """
    def __init__(self, args, logger, writer, device, generator):
        # 确保有target_domain参数
        if not hasattr(args, 'finetune_domain'):
            args.finetune_domain = "0"  # 默认为A域
        
        # 先设置target_domain，再调用父类初始化
        self.finetune_domain = args.finetune_domain
        # 获取域名的用户友好表示
        self.domain_name = "A" if self.finetune_domain == "0" else "B"
        
        # 现在调用父类初始化
        super().__init__(args, logger, writer, device, generator)
        
        # 设置迭代轮数和学习率
        self.epochs = args.domain_adapt_epochs if hasattr(args, 'domain_adapt_epochs') else 300
        self.lr = args.domain_adapt_lr if hasattr(args, 'domain_adapt_lr') else 1e-4
    
    def _create_model(self):
        """创建领域适配器模型"""
        self.user_num = self.generator.user_num
        self.item_num = self.generator.item_num
        self.item_num_dict = self.generator.get_item_num_dict()
        
        # 给args添加必要的属性
        self.args.user_num = self.user_num
        self.args.item_numA = self.item_num_dict["0"]
        self.args.item_numB = self.item_num_dict["1"]
        
        # 创建模型
        # Reset peak stats before model creation and moving to device to capture this phase accurately
        if self.device.type == 'cuda':
            # Ensure the device object passed is correctly interpreted by CUDA API
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}') 
            torch.cuda.reset_peak_memory_stats(device_for_stats) # Reset before loading model

        self.model = DomainSpecificAdapter(self.args)
        self.model.to(self.device)

        if self.device.type == 'cuda':
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}') 
            self.logger.info("--- GPU Memory After Model Creation and to(device) ---")
            allocated_mem_gb = torch.cuda.memory_allocated(device_for_stats) / (1024**3)
            peak_allocated_mem_gb = torch.cuda.max_memory_allocated(device_for_stats) / (1024**3)
            reserved_mem_gb = torch.cuda.memory_reserved(device_for_stats) / (1024**3)
            peak_reserved_mem_gb = torch.cuda.max_memory_reserved(device_for_stats) / (1024**3)
            self.logger.info(
                f"Model Load: 已分配 {allocated_mem_gb:.3f} GB (峰值 {peak_allocated_mem_gb:.3f} GB), "
                f"已预留 {reserved_mem_gb:.3f} GB (峰值 {peak_reserved_mem_gb:.3f} GB)"
            )
            self.logger.info("---------------------------------------------------------")
        
        # 确保这里使用的domain_name已经在__init__中设置好
        self.logger.info(f"创建领域适配器模型，目标域: {self.domain_name}")
    
    def _prepare_train_inputs(self, batch):
        """准备训练输入 - 与CDSRRegSeq2SeqDatasetUser返回值匹配"""
        inputs = {
            "seq": batch[0],          # 全域序列
            "pos": batch[1],          # 正样本
            "neg": batch[2],          # 负样本
            "positions": batch[3],    # 位置编码
            "seqA": batch[4],         # A域序列
            "posA": batch[5],         # A域正样本
            "negA": batch[6],         # A域负样本
            "positionsA": batch[7],   # A域位置编码
            "seqB": batch[8],         # B域序列
            "posB": batch[9],         # B域正样本
            "negB": batch[10],        # B域负样本
            "positionsB": batch[11],  # B域位置编码
            "target_domain": batch[12], # 目标域
            "domain_mask": batch[13], # 域掩码
            "reg_A": batch[14],       # A域正则化项
            "reg_B": batch[15],       # B域正则化项
            "user_id": batch[16]      # 用户ID
        }
        return inputs
    
    def _prepare_eval_inputs(self, batch):
        """准备评估输入 - 与CDSREvalSeq2SeqDataset返回值匹配"""
        # 注意：eval数据集使用的是CDSREvalSeq2SeqDataset
        inputs = {
            "seq": batch[0],           # 全域序列
            "positions": batch[3],     # 位置编码
            "seqA": batch[4],          # A域序列
            "positionsA": batch[7],    # A域位置编码
            "seqB": batch[8],          # B域序列
            "positionsB": batch[11],   # B域位置编码
            "item_indices": torch.cat([batch[1].unsqueeze(1), batch[2]], dim=1),  # 合并正负样本作为待评估物品
            # 以下字段用于评估，不传给模型的predict方法
            "pos": batch[1],           # 正样本
            "neg": batch[2],           # 负样本
            "posA": batch[5],          # A域正样本 
            "negA": batch[6],          # A域负样本
            "posB": batch[9],          # B域正样本
            "negB": batch[10],         # B域负样本
            "target_domain": batch[12] # 目标域
        }
        return inputs
    
    def train(self):
        """训练领域适配器模型"""
        self.logger.info(f"开始训练领域适配器，目标域: {self.domain_name}")
        
        # 准备训练和验证数据
        self.train_loader = self.generator.make_trainloader()
        self.valid_loader = self.generator.make_evalloader()
        self.test_loader = self.generator.make_evalloader(test=True)
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=0.01
        )
        
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 记录最佳性能
        best_performance = 0.0
        patience_counter = 0
        best_epoch = 0
        
        # <DEBUG COUNTER INIT START>
        if not hasattr(self.args, 'debug_prints_batch_counter'):
            self.args.debug_prints_batch_counter = 0
        # <DEBUG COUNTER INIT END>
        
        # 开始训练
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            epoch_start_time = time.time() # Start epoch timer
            total_batch_train_time_ms = 0.0 # Accumulator for batch training times

            if self.device.type == 'cuda':
                device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}') 
                if epoch > 1: # Only reset peak stats from the second epoch onwards
                    torch.cuda.reset_peak_memory_stats(device_for_stats)
            
            # <DEBUG COUNTER RESET FOR EPOCH START>
            # 如果您希望每个epoch的前几个批次都打印，请在此处重置计数器
            # 如果只想查看整个训练过程的最初几个批次，则注释掉下面这行
            self.args.debug_prints_batch_counter = 0 
            # <DEBUG COUNTER RESET FOR EPOCH END>
            
            # 训练一个epoch
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")):
                # <DEBUG COUNTER INCREMENT START>
                # 计数器在模型内部的条件判断之前增加，所以模型内用 < 3 判断会打印 batch 0, 1, 2
                self.args.debug_prints_batch_counter +=1 
                # <DEBUG COUNTER INCREMENT END>
                
                batch_start_time = time.time() # Start batch timer
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self._prepare_train_inputs(batch)
                
                loss = self.model(**inputs)
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                
                optimizer.step()
                
                train_loss += loss.item()
                batch_end_time = time.time() # End batch timer
                total_batch_train_time_ms += (batch_end_time - batch_start_time) * 1000
            
            # 计算平均损失和时间
            epoch_end_time = time.time()
            epoch_duration_ms = (epoch_end_time - epoch_start_time) * 1000
            avg_epoch_train_loss = train_loss / len(self.train_loader)
            
            self.logger.info(f"Epoch {epoch}, 平均训练损失: {avg_epoch_train_loss:.5f}")
            self.logger.info(f"Epoch {epoch}, 训练耗时: {epoch_duration_ms:.2f} ms")


            
            # 评估
            current_performance = self.eval(epoch)
            
            # 更新学习率
            scheduler.step(current_performance)
            
            # 判断是否为最佳模型
            if current_performance > best_performance:
                best_performance = current_performance
                best_epoch = epoch
                self.save_model()
                patience_counter = 0
                self.logger.info(f"Epoch {epoch}: 发现新的最佳模型，性能: {best_performance:.5f}")
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch}: 不是最佳模型，性能: {current_performance:.5f}")
            
            # 早停
            if patience_counter >= self.args.patience:
                self.logger.info(f"早停: {patience_counter} 个epoch没有改进")
                self.logger.info(f"最佳性能出现在 Epoch {best_epoch}: {best_performance:.5f}")
                break
        
        # 测试
        self.logger.info("训练完成，开始测试...")
        self.eval(test=True)
    
    def eval(self, epoch=0, test=False):
        """评估模型"""
        eval_start_time = time.time() # Start eval timer
        total_batch_eval_time_ms = 0.0

        # For eval, we are interested in the peak memory usage *during* the evaluation loop itself.
        # Model is already on GPU. reset_peak_memory_stats helps isolate eval loop's dynamic memory.
        if self.device.type == 'cuda':
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}') 
            torch.cuda.reset_peak_memory_stats(device_for_stats)

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info(f"********** 域 {self.domain_name} 模型测试 **********")
            desc = '测试中'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model_peft.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info(f"********** Epoch: {epoch} 域 {self.domain_name} 评估 **********")
            desc = '评估中'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)
        # 【修改2】添加seq_len收集（domain_adapter_trainer中缺少这个）
        # seq_len = torch.empty(0).to(self.device)

        domain_mask = torch.empty(0).to(self.device)
        
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            target_items = torch.cat([target_items, inputs["pos"]])
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            # 【修改3】在循环中收集seq_len
            # seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            
            
            with torch.no_grad():
                # inputs["pos"], inputs["neg"] 假设已经是全局ID (来自batch[1], batch[2])
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)

                # inputs["posA"], inputs["negA"] 是本地A域ID.
                # 假设A域本地ID就是其全局ID (0 to item_numA-1), 无需偏移.
                inputs["item_indicesA"] = torch.cat([inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)

                inputs["item_indicesB"] = torch.cat([inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)
                
                batch_model_predict_start_time = time.time() # Start timer for model.predict
                pred_logits = -self.model.predict(**inputs)
                batch_model_predict_end_time = time.time() # End timer for model.predict
                total_batch_eval_time_ms += (batch_model_predict_end_time - batch_model_predict_start_time) * 1000
                
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])
        
        eval_end_time = time.time()
        eval_duration_ms = (eval_end_time - eval_start_time) * 1000
        avg_batch_eval_time_ms = total_batch_eval_time_ms / len(test_loader)

        eval_phase_name = "测试" if test else f"Epoch {epoch} 评估"
        self.logger.info(f"{eval_phase_name} 总耗时: {eval_duration_ms:.2f} ms")
        self.logger.info(f"{eval_phase_name} 平均批次推理耗时: {avg_batch_eval_time_ms:.2f} ms/batch")

        if self.device.type == 'cuda':
            device_for_stats = torch.device(f'cuda:{torch.cuda.current_device()}') 
            # 'allocated_mem_gb' here includes model static memory + any remaining from last batch
            allocated_mem_gb = torch.cuda.memory_allocated(device_for_stats) / (1024**3)
            # 'peak_allocated_mem_gb' here is the peak *during this eval loop* due to reset_peak_memory_stats above
            peak_allocated_mem_gb = torch.cuda.max_memory_allocated(device_for_stats) / (1024**3)
            reserved_mem_gb = torch.cuda.memory_reserved(device_for_stats) / (1024**3)
            peak_reserved_mem_gb = torch.cuda.max_memory_reserved(device_for_stats) / (1024**3)
            self.logger.info(
                f"{eval_phase_name}, GPU显存: 当前已分配 {allocated_mem_gb:.3f} GB (评估/测试循环中峰值 {peak_allocated_mem_gb:.3f} GB), "
                f"当前已预留 {reserved_mem_gb:.3f} GB (评估/测试循环中峰值 {peak_reserved_mem_gb:.3f} GB)"
            )

        # 【修改4】获取item流行度字典
        # 假设generator有get_item_pop_dict方法，或者从数据集中获取
        # item_pop_dict = self.generator.get_item_pop_dict()  # 需要在generator中实现这个方法
        # 或者从其他地方加载预计算的流行度
        # 如果没有现成的，可以暂时创建一个dummy dict用于测试
        # item_pop_dict = {i: np.random.randint(1, 200) for i in range(1, self.item_num+1)}

        # 获取整体性能
        self.logger.info('')
        
        # 【修改5】替换原来的metric_report调用
        # 原来的代码：
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        
        # 【修改6】新的5组评估代码（替换上面的res_dict = metric_report行）：
        # pred_rank_np = pred_rank.detach().cpu().numpy()
        # seq_len_np = seq_len.detach().cpu().numpy()
        # target_items_np = target_items.detach().cpu().numpy()
        
        # # 按序列长度分组评估
        # HR_len, NDCG_len, count_len = metric_len_5group(pred_rank_np, seq_len_np, 
        #                                                 thresholds=[5, 10, 15, 20], topk=10)
        
        # # 按物品流行度分组评估
        # HR_pop, NDCG_pop, count_pop = metric_pop_5group(pred_rank_np, item_pop_dict, target_items_np,
        #                                                 thresholds=[10, 30, 60, 100], topk=10)
        
        # # 构建详细的res_dict
        # res_dict = {}
        # # 添加原始的整体指标
        # overall_metrics = metric_report(pred_rank_np)
        # res_dict.update(overall_metrics)
        
        # # 添加按序列长度分组的指标
        # len_groups = ['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
        # for i, group in enumerate(len_groups):
        #     res_dict[f'HR@10_len_{group}'] = HR_len[i]
        #     res_dict[f'NDCG@10_len_{group}'] = NDCG_len[i]
        
        # # 添加按流行度分组的指标
        # pop_groups = ['Tail', 'Unpopular', 'Medium', 'Popular', 'Head']
        # for i, group in enumerate(pop_groups):
        #     res_dict[f'HR@10_pop_{group}'] = HR_pop[i]
        #     res_dict[f'NDCG@10_pop_{group}'] = NDCG_pop[i]

        #res_dict_pop = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_num_dict, target_items)
        #res_dict_len = metric_len_report(pred_rank.detach().cpu().numpy(), target_domain)
        # 区分域A和域B的性能
        pred_rank_A = pred_rank[target_domain == 0]
        pred_rank_B = pred_rank[target_domain == 1]
        
        # 【修改7】为域A和域B也应用5组评估
        # 原来的代码：
        res_dict_A = metric_report(pred_rank_A.detach().cpu().numpy())
        res_dict_B = metric_report(pred_rank_B.detach().cpu().numpy())
        
        # 【修改8】新的域A、B的5组评估代码（替换上面两行）：
        # # 域A的详细评估
        # domain_A_mask = target_domain == 0
        # pred_rank_A_np = pred_rank_A.detach().cpu().numpy()
        # seq_len_A_np = seq_len[domain_A_mask].detach().cpu().numpy()
        # target_items_A_np = target_items[domain_A_mask].detach().cpu().numpy()
        
        # HR_len_A, NDCG_len_A, count_len_A = metric_len_5group(pred_rank_A_np, seq_len_A_np, 
        #                                                       thresholds=[5, 10, 15, 20], topk=10)
        # HR_pop_A, NDCG_pop_A, count_pop_A = metric_pop_5group(pred_rank_A_np, item_pop_dict, target_items_A_np,
        #                                                       thresholds=[10, 30, 60, 100], topk=10)
        
        # res_dict_A = metric_report(pred_rank_A_np)
        # for i, group in enumerate(len_groups):
        #     res_dict_A[f'HR@10_len_{group}'] = HR_len_A[i]
        #     res_dict_A[f'NDCG@10_len_{group}'] = NDCG_len_A[i]
        # for i, group in enumerate(pop_groups):
        #     res_dict_A[f'HR@10_pop_{group}'] = HR_pop_A[i]
        #     res_dict_A[f'NDCG@10_pop_{group}'] = NDCG_pop_A[i]
        
        # # 域B的详细评估
        # domain_B_mask = target_domain == 1
        # pred_rank_B_np = pred_rank_B.detach().cpu().numpy()
        # seq_len_B_np = seq_len[domain_B_mask].detach().cpu().numpy()
        # target_items_B_np = target_items[domain_B_mask].detach().cpu().numpy()
        
        # HR_len_B, NDCG_len_B, count_len_B = metric_len_5group(pred_rank_B_np, seq_len_B_np, 
        #                                                       thresholds=[5, 10, 15, 20], topk=10)
        # HR_pop_B, NDCG_pop_B, count_pop_B = metric_pop_5group(pred_rank_B_np, item_pop_dict, target_items_B_np,
        #                                                       thresholds=[10, 30, 60, 100], topk=10)
        
        # res_dict_B = metric_report(pred_rank_B_np)
        # for i, group in enumerate(len_groups):
        #     res_dict_B[f'HR@10_len_{group}'] = HR_len_B[i]
        #     res_dict_B[f'NDCG@10_len_{group}'] = NDCG_len_B[i]
        # for i, group in enumerate(pop_groups):
        #     res_dict_B[f'HR@10_pop_{group}'] = HR_pop_B[i]
        #     res_dict_B[f'NDCG@10_pop_{group}'] = NDCG_pop_B[i]
        
        # 输出整体性能
        self.logger.info("整体性能:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        # 输出域A性能
        self.logger.info("域A性能:")
        for k, v in res_dict_A.items():
            if not test:
                self.writer.add_scalar('Test/A_{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        # 输出域B性能
        self.logger.info("域B性能:")
        for k, v in res_dict_B.items():
            if not test:
                self.writer.add_scalar('Test/B_{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        # 合并性能指标
        res_dict = {**res_dict, 
                   **{f"A_{k}": v for k, v in res_dict_A.items()},
                   **{f"B_{k}": v for k, v in res_dict_B.items()}}
        
        if test:
            record_csv(self.args, res_dict)
        
        # 根据目标域返回性能
        #if self.finetune_domain == "0":
        #    return res_dict_A.get("NDCG@10", 0.0)
        #else:
        #    return res_dict_B.get("NDCG@10", 0.0)
        return res_dict.get("NDCG@10", 0.0)
    def save_model(self):
        """保存模型"""
        model_to_save = self.model
        output_model_file = os.path.join(self.args.output_dir, "pytorch_model_peft.bin")
        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        torch.save({'state_dict': model_to_save.state_dict()}, output_model_file)
        self.logger.info(f"模型保存到 {output_model_file}")
