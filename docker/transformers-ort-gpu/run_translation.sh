#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/stage/transformers/src
export USE_TF=0

BS=2
python3 run_seq2seq.py \
        --source_prefix "translate English to  Romanian: " \
        --dataset_name wmt16 \
        --dataset_config "ro-en" \
        --model_name_or_path t5-large \
        --output_dir output_dir \
        --adam_eps 1e-06 \
        --do_train \
        --label_smoothing 0.1 \
        --learning_rate 3e-5 \
        --logging_first_step \
        --logging_steps 1000 \
        --max_source_length 128 \
        --max_target_length 128 \
        --num_train_epochs 1 \
        --overwrite_output_dir \
        --per_device_train_batch_size $BS \
        --predict_with_generate \
        --sortish_sampler \
        --task translation_en_to_ro \
        --warmup_steps 5 \
        --max_train_samples 1024 \
        --fp16 \
        #--ortmodule

