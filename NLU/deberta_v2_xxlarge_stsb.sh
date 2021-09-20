export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./stsb"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--lora_path mnli/pytorch_model_lora.bin \
--task_name stsb \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 4 \
--learning_rate 2e-4 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 50 \
--save_strategy steps \
--save_steps 50 \
--warmup_steps 100 \
--cls_dropout 0.2 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.1 \
--use_deterministic_algorithms
