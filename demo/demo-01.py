from peft import get_peft_model, LoraConfig, TaskType

# 设置超参数及配置
from transformers import AutoModelForSeq2SeqLM

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

TARGET_MODULES = ["q_proj", "v_proj", ]

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# 创建基础TRANSFORMER模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# 加入PEFT策略
model = get_peft_model(model, config)
