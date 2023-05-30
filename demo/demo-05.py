from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

import torch

data = pd.read_excel("historical_data.xlsx", sheet_name=0).fillna(" ")

data['text'] = data['title'].apply(lambda x: str(x) if x else "") + data['content'].apply(lambda x: str(x) if x else "")

model_name = "bert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    device = "cuda:0"
    model.half()
else:
    device = "cpu"

model = model.to(device)

max_target_length = 128

label2id = {
    '招聘信息': 0,
    '经验贴': 1,
    '求助贴': 2
}
id2label = {v: k for k, v in label2id.items()}


def get_answer(text):
    #
    text = [x for x in text]

    inputs = tokenizer(text, return_tensors="pt", max_length=max_target_length, padding=True, truncation=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        #
        outputs = model(**inputs).logits.argmax(-1).tolist()

    return outputs


# print(get_answer(data['text'][:10]))

pred, grod = [], []

index, batch_size = 0, 32

while index < len(data['text']):
    #
    pred.extend(get_answer([x for x in data['text'][index:index + batch_size]]))

    index += batch_size

# print(pred)
# print(grod)

pred = [id2label[x] for x in pred]

data["target"] = pred

writer = pd.ExcelWriter("generate.xlsx")

data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')

writer.save()
writer.close()
