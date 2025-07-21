# domain_adapter.bash
gpu_id=0
dataset="food_kitchen"
#seed_list=(42 44 46)
seed_list=(42)
#inter_file="food_kitchen"
#llm_emb_file="itm_emb_np"
inter_file="aug_food_kitchen"
llm_emb_file="aug_itm_emb_np"
user_emb_file="domain_split_usr_profile_emb"


model_name="DomainSpecificAdapter"
finetune_domain="0"  # 指定目标域

# 基础参数
hidden_size=128
hard_negative_weight=0.1

alpha=1.0
beta=0.5
gamma=0.3
delta=0.7
# 适配器参数
adapter_size=32
domain_adapt_epochs=300
domain_adapt_lr=1e-4
l2_weight=0.01
num_heads=1

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
                --num_train_epochs 200 \
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