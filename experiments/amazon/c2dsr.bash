## C2DSR for Cross-Domain Sequential Recommendation - Amazon Dataset
gpu_id=0
dataset="amazon"
seed_list=(42)
inter_file="cloth_sport"
llm_emb_file="itm_emb_np"
user_emb_file="raw_usr_profile_emb"

model_name="C2DSR"
# C2DSR特定参数 - Amazon数据集调优
alpha=0.1
beta=0.1
mi_weight=0.05  # 互信息损失权重（Amazon数据集较小值）
domain_weight=0.3  # 域特定增强权重

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
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "c2dsr_amazon" \
                --patience 20 \
                --ts_user 12 \
                --ts_item 13 \
                --log \
                --domain "AB" \
                --trm_num 2 \
                --alpha ${alpha} \
                --beta ${beta}
done 