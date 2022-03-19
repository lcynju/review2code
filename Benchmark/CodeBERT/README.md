# CodeBERT
 [https://github.com/microsoft/CodeBERT](https://github.com/microsoft/CodeBERT)
## Fine-Tune
We fine-tuned the model on 4*V100 GPUs. 
```shell
lang=java #fine-tuning a language-specific model for each programming language 
pretrained_model=/pretrain_model/codebert-base  #Roberta: roberta-base
nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file k9mail_train_neg.json \
--dev_file k9mail_valid_neg.json \
--max_seq_length 512 \
--per_gpu_train_batch_size 12 \
--per_gpu_eval_batch_size 12 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ./data/ \
--output_dir ./models/k9mail \
--model_name_or_path $pretrained_model > train_k9mail.txt 2>&1 &

```
## Test

Inference
```shell
python test.py
```
## Requirements

- python 3.9.7 
- pytorch 1.10.1
- transformer 4.15.0
- tree-sitter 0.20.0

