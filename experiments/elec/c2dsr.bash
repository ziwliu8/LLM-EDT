## C2DSR for Cross-Domain Sequential Recommendation
gpu_id=0
dataset="elec"
seed_list=(42)
inter_file="elec_phone"
llm_emb_file="itm_emb_np"
user_emb_file="usr_profile_emb"

model_name="C2DSR"
# C2DSR特定参数
alpha=1.0
beta=1.0
mi_weight=0.1  # 互信息损失权重
domain_weight=0.5  # 域特定增强权重

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
                --check_path "c2dsr_elec" \
                --patience 20 \
                --ts_user 12 \
                --ts_item 13 \
                --log \
                --domain "AB" \
                --trm_num 2 \
                --num_heads 1 \
                --dropout_rate 0.1 \
                --learning_rate 0.001 \
                --weight_decay 5e-4 \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                --alpha ${alpha} \
                --beta ${beta}
done 