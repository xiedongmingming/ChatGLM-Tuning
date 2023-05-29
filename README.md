# CHATGLM-TUNING

一种平价的CHATGPT实现方案，基于清华的[CHATGLM-6B](https://github.com/THUDM/ChatGLM-6B)+LORA进行FINETUNE。

数据集：[alpaca](https://github.com/tatsu-lab/stanford_alpaca)

有COLAB的同学可以直接在COLAB上尝试：<a href="https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/finetune.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

[官方PTUNING代码](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning)


## DEMO

- [开源版的文心一言](https://github.com/visual-openllm/visual-openllm)


## S1 FINETUNE

### 准备

- 显卡：显存>=16G(最好24G或者以上)
- 环境：
- - PYTHON>=3.8
- - CUDA>=11.6，CUPTI，CUDNN，TENSORRT等深度学习环境
- - pip3 install -r requirements.txt


### 数据预处理


转化ALPACA数据集为JSONL

```bash
python cover_alpaca2jsonl.py \
    --data_path data/alpaca_data.json \
    --save_path data/alpaca_data.jsonl \
```

tokenization

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 200 \ 
    --skip_overlength
```

- `--jsonl_path`：微调的数据路径，格式JSONL，对每行的['CONTEXT']和['TARGET']字段进行ENCODE。
- `--save_path`：输出路径。
- `--max_seq_length`：样本的最大长度。

### 训练

```bash
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
```

### 推理

参考[INFER.IPYNB](infer.ipynb)

<details><summary><b>FINETUNE前后对比</b></summary>


利用ALPACA数据集合对CHATGLM-6B FINETUNE后，在ALPACA数据集上表现得更好：
- `ANSWER:` 是模型的输出
- `#### ANSWER:` 是原答案
![](https://user-images.githubusercontent.com/6883957/226977555-c00c796f-4fdb-4613-810a-8b9a6068bb1b.jpeg)


</details>


## S2. REWARD MODEL

## S3. PPO


## LORA

| LORA                                  | DATASET      |
| ------------------------------------- | ------------ |
| mymusise/chatglm-6b-alpaca-lora       | Alpaca       |
| mymusise/chatglm-6b-alpaca-zh-en-lora | Alpaca-zh-en |
| *(on the way)*                        | Alpaca-zh    |

### 使用预训练好的LORA

参考[examples/infer_pretrain.ipynb](https://colab.research.google.com/github/mymusise/ChatGLM-Tuning/blob/master/examples/infer_pretrain.ipynb)


# TODO:

- ~~bs > 1 support~~
- 使用中文数据
- 加入RLHF