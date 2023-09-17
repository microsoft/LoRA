python examples/text-classification/run_glue.py \
--model_name_or_path textattack/roberta-base-SST-2 \
--output_dir ./output \
--task_name sst2 \
--do_eval \
--apply_lora \
--lora_r 128 \
--lora_alpha 128 \
--seed 0 \
--ex_type dW=WSVD
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/lora/2023-09-11_16:37:03lorackpt.bin
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16r_no_merged/model/checkpoint-15810/pytorch_model.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/lora/2023-09-17_17:47_lorackpt.bin \