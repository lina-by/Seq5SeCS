import numpy as np
import pandas as pd
import numpy as np
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch
import torch.nn.functional as F
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# name = "tasksource/ModernBERT-large-nli"
name = "sileod/deberta-v3-base-tasksource-nli"
nli_model = pipeline(
        "zero-shot-classification",
        model=name,
        device=device,
        batch_size=128
    )

with open('data/test_without_labels.txt', 'r') as file:
    data = file.readlines()

data = [x.replace('\n', '') for x in data]
d = {'texts': data} 
test_df = pd.DataFrame(d)
test_df = test_df.dropna()
data = test_df["texts"].tolist()
candidate_labels = ['Sports','Politics','Health','Finance','Travel','Food','Education','Environment','Fashion','Science','Technology','Entertainment']


labels = nli_model(data, candidate_labels)
y_pred = []
for i in range(len(labels)):
    label = labels[i]['labels']
    prob = labels[i]['scores']
    arg = np.argmax(prob)
    y_pred.append(label[arg])
    
with open('y_pred_shuffle.txt', 'w') as file:
    for item in y_pred:
        file.write(item + "\n")