#!/bin/bash -ex


trials=squad_ort_fp16_3
mkdir -p test_result2/$trials/tensorboard
python -m torch.distributed.launch --nproc_per_node=4 ./run_qa.py \
  --cache_dir "./cache_dir" \
  --model_name_or_path roberta-large \
  --dataset_name squad \
  --do_train --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 256 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir test_result2/$trials --overwrite_output_dir \
  --logging_dir test_result2/$trials/tensorboard --logging_first_step --logging_steps 50 \
  --fp16 \
  --ort \
  --deepspeed ds_config.json
  2>&1 | tee test_result2/$trials/log.txt
