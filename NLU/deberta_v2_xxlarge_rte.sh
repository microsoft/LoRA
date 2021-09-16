export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./rte"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--lora_path mnli/pytorch_model_lora.bin \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 4 \
--learning_rate 2.6e-4 \
--num_train_epochs 11 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 20 \
--save_strategy steps \
--save_steps 20 \
--warmup_steps 50 \
--cls_dropout 0.2 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--use_deterministic_algorithms
