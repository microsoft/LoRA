export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./adapter_houlsby_roberta_large_mnli"
python -m torch.distributed.launch --nproc_per_node=num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--task_name mnli \
--do_train \
--do_eval \
--evaluation_strategy epoch \
--save_strategy epoch \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 3e-4 \
--num_train_epochs 5 \
--output_dir output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir output_dir/log \
--warmup_ratio 0.1 \
--apply_adapter \
--adapter_type pfeiffer \
--adapter_size 16 \
--seed 0
