CUDA_VISIBLE_DEVICES=0 python -u ./inference.py --tokenizer_path "canho/koalpaca-5.8b-3epochs-30000-data" \
                                                       --checkpoint_idx 1044 --dataset_path "dataset/dpo_train.csv" \
                                                       --save_folder "../inference_output/"