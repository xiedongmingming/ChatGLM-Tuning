import pandas as pd
import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from transformers import EvalPrediction

from transformers import logging

######################################################################################
# 设置日志

logging.set_verbosity_info()
# 通过环境变量TRANSFORMERS_VERBOSITY：debug, info, warning, error, critical
# TRANSFORMERS_NO_ADVISORY_WARNINGS
######################################################################################
model_name = "uer/chinese_roberta_L-4_H-512"

tokenizer = AutoTokenizer.from_pretrained(model_name)

label2id = {
    '招聘信息': 0,
    '经验贴': 1,
    '求助贴': 2
}
id2label = {v: k for k, v in label2id.items()}

max_input_length = 128


def preprocess_function(examples):
    #
    # examples：text+target
    #
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, truncation=True)

    labels = [label2id[x] for x in examples['target']]

    model_inputs["labels"] = labels

    return model_inputs  # input_ids、token_type_ids、attention_mask、labels


raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names  # 移除CSV结构中的标题头
)

######################################################################################
# 定义评价指标函数
#
#   评价指标METRIC用于EVALUATE的时候衡量模型的表现，这里使用F1-SCORE和ACCURACY
#
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

import evaluate

metric = evaluate.load("seqeval")  # TODO ？？？评估函数


def multi_label_metrics(predictions, labels, threshold=0.5):
    # (105,3),(105,)
    probs = np.argmax(predictions, -1)  # 获取最大值索引

    y_true = labels

    f1_micro_average = f1_score(y_true=y_true, y_pred=probs, average='micro')

    accuracy = accuracy_score(y_true, probs)

    print(classification_report([id2label[x] for x in y_true], [id2label[x] for x in probs]))  # 生成报告

    # return as dictionary
    metrics = {
        'f1': f1_micro_average,
        'accuracy': accuracy
    }

    return metrics


def compute_metrics(p: EvalPrediction):
    #
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions  # ndarray：三个类别分别的概率(105,3)

    result = multi_label_metrics(predictions=preds, labels=p.label_ids)  # (105,)

    return result


######################################################################################
# 指定模型的训练参数
#
#       加载模型，并构建TrainingArguments类，用于指定模型训练的各种参数
#       第一个是训练保存地址为必填项，其他都是选填项
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    # problem_type="multi_label_classification",
    num_labels=3,
    # id2label=id2label,
    # label2id=label2id
)

batch_size = 64

metric_name = "f1"

training_args = TrainingArguments(
    f"F:\\workspace\\notebook\\ChatGLM-Tuning\\demo\\output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # gradient_accumulation_steps=2,
    num_train_epochs=10,
    save_total_limit=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    fp16=True,
)
# 定义TRAINER并进行训练
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # 这里是自定义METRIC的地方
)

trainer.train()  # 开始训练

# 测试预测
print("test")
print(trainer.evaluate())  # 测试

trainer.save_model("bert")  # 保存模型

# 进行模型预测，并将预测结果输出便于观察
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])

predictions = np.argmax(predictions, axis=-1)

print(predictions)
print(labels)
