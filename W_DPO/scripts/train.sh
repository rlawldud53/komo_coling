CUDA_VISIBLE_DEVICES=0 python -u ./train.py \
model=blank_model \
model.tokenizer_name_or_path="beomi/KoAlpaca-Polyglot-5.8B" \
model.name_or_path="beomi/KoAlpaca-Polyglot-5.8B" \
datasets=['custom'] exp_name="koalpaca datasets only for dpo" \
trainer=BasicTrainer sample_during_eval=false loss.label_smoothing=1 loss.beta=2 loss.reference_free=False \
local_run_dir="./config" eval_every=350 \
exp_name="0th_trial_40000_data" 