#!/bin/bash -ex

export BS=16 CUDA_VISIBLE_DEVICES=1,2,0,3 USE_TF=0

result_dir=res/${1}_${2}_${3}
mkdir -p ${result_dir}

python -m torch.distributed.launch --nproc_per_node=${1} ./run_qa.py \
  --model_name_or_path roberta-large \
  --dataset_name adversarial_qa --dataset_config "droberta" \
  --output_dir $result_dir --overwrite_output_dir \
  --logging_dir $result_dir/tensorboard --logging_first_step --logging_steps 50 \
  --do_train --learning_rate 1e-5 --adam_eps 1e-06 --label_smoothing 0.1 \
  --num_train_epochs 1 --max_steps=1000 \
  --per_device_eval_batch_size $BS --per_device_train_batch_size $BS \
  --warmup_steps 2000 \
  --dataloader_drop_last \
  ${2} \
  ${3} \
  2>&1 | tee $result_dir/log_${1}_${2}_${3}.txt
  #--do_eval --evaluation_strategy=steps --eval_steps 1000 \

