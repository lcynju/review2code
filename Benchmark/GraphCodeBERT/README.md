# GraphCodeBERT
[https://github.com/microsoft/CodeBERT](https://github.com/microsoft/CodeBERT)
 
## Fine-Tune
We fine-tuned the model on 4*V100 GPUs. 
```shell
lang=java
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/ \
    --config_name=./pretrained_model/graphcodebert-base \
    --model_name_or_path=./pretrained_model/graphcodebert-base \
    --tokenizer_name=./pretrained_model/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=data/k9mail_train.json \
    --eval_data_file=data/k9mail_valid.json \
    --codebase_file=data/k9mail_code.json \
    --num_train_epochs 30 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 1234567 2>&1| tee saved_models/train_k9mail.logi


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
