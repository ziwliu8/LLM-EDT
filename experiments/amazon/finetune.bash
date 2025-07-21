gpu_id=0
dataset="amazon"
seed_list=(42)
inter_file="aug_cloth_sport"
llm_emb_file="aug_itm_emb_np"
user_emb_file="aug_usr_profile_emb"

model_name="One4All_Finetuner"  # 使用域增强微调模型
target_domain="0"  # 0表示A域，1表示B域
pretrained_model_path="./saved/amazon/One4All/pytorch_model0.bin"  # 预训练模型路径

# 微调参数
lambda_pos=1.0  # 正样本权重
lambda_neg=0.3  # 负样本权重
tau=0.1  # 对比损失温度参数

for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --inter_file ${inter_file} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 4 \
                --num_train_epochs 100 \
                --seed ${seed} \
                --patience 20 \
                --log \
                --domain "AB" \
                --global_emb \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                --target_domain ${target_domain} \
                --pretrained_model_path ${pretrained_model_path} \
                --lambda_pos ${lambda_pos} \
                --lambda_neg ${lambda_neg} \
                --tau ${tau} 
done 