import pandas as pd
import torch
from transformers import pipeline

test_data = pd.read_csv("test_without_labels.csv", encoding="utf-8")
test_data = test_data.dropna()
sentences = test_data["Text"].tolist()

device = 0 if torch.cuda.is_available() else -1
model_ckpt = "xlm-roberta-base-finetuned-language-detection/checkpoint-2440"
model_pipe = pipeline(
    "text-classification", model=model_ckpt, device=device, batch_size=64
)
labels = model_pipe(sentences, truncation=True, max_length=128)

y_pred = []
for i in range(len(labels)):
    label = labels[i]["label"]
    y_pred.append({"ID": i + 1, "Label": label})

y_df = pd.DataFrame(y_pred)
y_df.to_csv("y_pred_lang.csv", index=False)
