# domain_adapter.bash
gpu_id=0
dataset="amazon"
#seed_list=(42 44 46)
seed_list=(42)
#seed_list=(44 46)
inter_file="aug_cloth_sport"
llm_emb_file="aug_itm_emb_np"
#inter_file="cloth_sport"
#llm_emb_file="itm_emb_np"
user_emb_file="domain_split_usr_profile_emb"

#cold start
#inter_file="cold_cloth_sport"
#llm_emb_file="cold_itm_emb_np"
#inter_file="aug_cold_cloth_sport"
#llm_emb_file="aug_cold_itm_emb_np"

model_name="DomainSpecificAdapter"
finetune_domain="0"  # 指定目标域

# 基础参数
hidden_size=128

alpha=0.3
beta=0.7
gamma=0.3
delta=0.7
# 适配器参数
adapter_size=64
domain_adapt_epochs=300
domain_adapt_lr=1e-4
l2_weight=0.01
num_heads=2

for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --inter_file ${inter_file} \
                --model_name ${model_name} \
                --hidden_size ${hidden_size} \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 4 \
                --num_train_epochs 300 \
                --seed ${seed} \
                --check_path "domain_${finetune_domain}" \
                --patience 20 \
                --log \
                --domain "AB" \
                --alpha ${alpha} \
                --beta ${beta} \
                --gamma ${gamma} \
                --delta ${delta} \
                --global_emb \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                --augmented \
                --finetune_domain ${finetune_domain} \
                --adapter_size ${adapter_size} \
                --domain_adapt_epochs ${domain_adapt_epochs} \
                --domain_adapt_lr ${domain_adapt_lr} \
                --l2_weight ${l2_weight} \
                --num_heads ${num_heads} \

done