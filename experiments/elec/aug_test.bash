gpu_id=0
dataset="elec"
#seed_list=(42 44 46)
seed_list=(42)
#inter_file="elec_phone"
#llm_emb_file="itm_emb_np"
inter_file="aug_elec_phone"
llm_emb_file="aug_itm_emb_np"
user_emb_file="usr_profile_emb"

model_name="One4All"
alpha=1.0
beta=1.0
gamma=0.3
delta=0.7
hidden_size=128
hard_negative_weight=0.1
domain_beta=2.0
adaptive_weight=True

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
                --check_path "" \
                --patience 35 \
                --log \
                --domain "AB" \
                --global_emb \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                --alpha ${alpha} \
                --beta ${beta} \
                --gamma ${gamma} \
                --delta ${delta} \
                --augmented \

done