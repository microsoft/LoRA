export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./mnli"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 8 \
--learning_rate 1e-4 \
--num_train_epochs 5 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 1000 \
--cls_dropout 0.15 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--reg_loss_wgt 1.0 \
--seed 42 \
--weight_decay 0 \
--use_deterministic_algorithms
