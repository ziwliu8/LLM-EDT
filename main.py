# here put the import lib
import os
import argparse
import torch

from generators.generator import CDSRRegSeq2SeqGeneratorUser
from generators.aug_generator import Aug_CDSRRegSeq2SeqGeneratorUser
from trainers.cdsr_trainer import CDSRTrainer
#from trainers.cold_cdsr_trainer import CDSRTrainer
from trainers.domain_adapter_trainer import DomainAdapterTrainer
from utils.utils import set_seed
from utils.logger import Logger
from utils.argument import *
from trainers.lora_trainer import LoRATrainer

import setproctitle
setproctitle.setproctitle("LLM4CDSR")


parser = argparse.ArgumentParser()
parser = get_main_arguments(parser)
parser = get_model_arguments(parser)
parser = get_train_arguments(parser)
parser.add_argument('--augmented', action='store_true', help='Whether to use augmented generator')
parser.add_argument('--domain_beta', type=float, default=2.0, help='domain_beta')
parser.add_argument('--do_peft', action='store_true', help='Whether to use PEFT')
parser.add_argument('--peft_type', type=str, default='adapter', 
                   choices=['prompt', 'adapter', 'lora', 'interest_transfer'], 
                   help='Type of PEFT to use')
parser.add_argument('--adapter_size', type=int, default=64, 
                   help='Size of adapter hidden layer')
parser.add_argument('--peft_epochs', type=int, default=100, 
                   help='Number of epochs for PEFT training')
parser.add_argument('--peft_lr', type=float, default=1e-4, 
                   help='Learning rate for PEFT training')
parser.add_argument('--lora_rank', type=int, default=64, 
                   help='Rank for LoRA decomposition')
parser.add_argument('--lora_alpha', type=int, default=16, 
                   help='Alpha scaling factor for LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.1, 
                   help='Dropout rate for LoRA layers')
parser.add_argument('--lora_lr', type=float, default=1e-4, 
                   help='Learning rate for LoRA training')
parser.add_argument('--finetune_domain', type=str, default='0', choices=['0', '1'], 
                   help='Target domain for PEFT')
parser.add_argument('--pretrain_path', type=str, default='One4All',
                   help='Path to pretrained model')
torch.autograd.set_detect_anomaly(True)
parser.add_argument('--hard_negative_weight', type=float, default=0.1,
                   help='硬负样本的权重')
parser.add_argument('--adaptive_weight', type=bool, default=True,
                   help='是否使用自适应权重')
parser.add_argument('--l2_weight', type=float, default=0.01,
                   help='L2正则化权重')
parser.add_argument('--align_weight', type=float, default=0.1,
                   help='对齐权重')
parser.add_argument('--kd_weight', type=float, default=0.1,
                   help='知识蒸馏权重')
args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_path = os.path.join(args.output_dir, args.pretrain_path)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path
args.llm_emb_path = os.path.join("data/"+args.dataset+"/handled/", "{}.pkl".format(args.llm_emb_file))


def main():
    torch.cuda.empty_cache()
    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ["llm4cdsr", "One4All", "C2DSR", "One4AllAttentionOnly", "One4AllEmbeddingOnly", "DomainSpecificAdapter"]:
        if args.augmented:
            generator = Aug_CDSRRegSeq2SeqGeneratorUser(args, logger, device)
        else:
            generator = CDSRRegSeq2SeqGeneratorUser(args, logger, device)
    else:
        raise ValueError

    if args.model_name in ["llm4cdsr", "One4All", "C2DSR", "One4AllAttentionOnly", "One4AllEmbeddingOnly"]:
        if args.do_peft:
            if args.peft_type == 'lora':
                trainer = LoRATrainer(args, logger, writer, device, generator)
            else:
                trainer = CDSRTrainer(args, logger, writer, device, generator)
        else:
            trainer = CDSRTrainer(args, logger, writer, device, generator)
    elif args.model_name == "DomainSpecificAdapter":
        from trainers.domain_adapter_trainer import DomainAdapterTrainer
        trainer = DomainAdapterTrainer(args, logger, writer, device, generator)
        #trainer = CDSRTrainer(args, logger, writer, device, generator)
    else:
        raise ValueError

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_item_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



