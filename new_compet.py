from tqdm import tqdm
from fast_langdetect import detect
import pandas as pd
import numpy as np
import fasttext
import re

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) # returns top 2 matching languages
        return predictions

model = LanguageIdentification()

def normalize_text(line):
    line = re.sub(r"'", " ' ", line)
    line = re.sub(r'"', "", line)
    line = re.sub(r'\.', ' . ', line)
    line = re.sub(r'<br\s*/>', ' ', line)
    line = re.sub(r',', ' , ', line)
    line = re.sub(r'\(', ' ( ', line)
    line = re.sub(r'\)', ' ) ', line)
    line = re.sub(r'\!', ' ! ', line)
    line = re.sub(r'\?', ' ? ', line)
    line = re.sub(r'\;', ' ', line)
    line = re.sub(r'\:', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    return line

train_data = pd.read_csv("data/train_submission.csv", encoding="utf-8")
lines = train_data["Text"]
res = []
for line in tqdm(lines):
    line = normalize_text(line)
    try:
        # lang = detect(line)['lang']
        lang = model.predict_lang(line)
        lang = lang[0][0].replace('__label__', '')
    except:
        lang = "unknown"
    res.append(lang)

#Compare with train_data["Label"]
acc = 0
for i in range(len(train_data)):
    if train_data["Label"][i] == res[i]:
        acc += 1
print(acc/len(train_data))

#get distribution of languages
from matplotlib import pyplot as plt

labels = np.array(train_data["Label"].dropna())
labels, counts = np.unique(labels, return_counts=True)
#print top 30 most common
idx_sort = np.argsort(counts)[::-1]
labels = labels[idx_sort][:200]
counts = counts[idx_sort][:200]
plt.bar(labels, counts)
plt.show()

# test_data = pd.read_csv("data/test_without_labels.csv", encoding="utf-8")
# lines = test_data["Text"]
# res = []
# for line in tqdm(lines):
#     try:
#         lang = detect(line)['lang']
#     except:
#         lang = "unknown"
#     res.append(lang)

# test_data["Label"] = res
# test_data["ID"] = [i for i in range(len(test_data))]
# test_data.to_csv("test_with_labels.csv", index=False)