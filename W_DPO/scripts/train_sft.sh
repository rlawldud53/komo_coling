CUDA_VISIBLE_DEVICES=0 python -u ./train.py \
model.tokenizer_name_or_path="PrunaAI/saltlux-Ko-Llama3-Luxia-8B-bnb-4bit-smashed" \
model.name_or_path="PrunaAI/saltlux-Ko-Llama3-Luxia-8B-bnb-4bit-smashed" \
model=blank_model \
loss=sft exp_name="llama sft" \
batch_size=4 \
trainer=BasicTrainer sample_during_eval=false \
local_run_dir="./config" eval_every=350 \
exp_name="0th_trial_40000_data" \
datasets=['custom'] 