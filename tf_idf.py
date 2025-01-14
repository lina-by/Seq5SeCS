# import json

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

# import re
# import string

# import nltk
# from sklearn.utils import shuffle
import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# import nltk
# nltk.download('stopwords')

# stop_words = stopwords.words('english')

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report

# import matplotlib.pyplot as plt
# import seaborn as sns



# f_data = {}
# for k, v in data.items():
#     f_data[k] = v[:95]

# processed_data = {
#     'text': [], 'target': []
# }
# for k, v in f_data.items():
#     processed_data['text'] += v
#     processed_data['target'] += [k] * len(v)

# df = pd.DataFrame.from_dict(processed_data)

# X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['target'].values, test_size=0,
#                                                    random_state=123, stratify=df['target'].values)

# X_train = df['text'].values
# y_train = df['target'].values
# X_train, y_train = shuffle(X_train, y_train)
# with open('test_shuffle.txt', 'w') as file:
#     for item in X_train.tolist():
#         file.write(item + "\n")

# with open('y_test_shuffle.txt', 'w') as file:
#     for item in y_train.tolist():
#         file.write(item + "\n")
# y_test = y_train
# X_test = X_train
# tfidf_vectorizer = TfidfVectorizer()

# tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

# tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

# classifier = RandomForestClassifier()

# classifier.fit(tfidf_train_vectors, y_train)

# y_pred = classifier.predict(tfidf_test_vectors)

import torch
import torch.nn.functional as F
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
nli_model = pipeline(
        "zero-shot-classification",
        model="sileod/deberta-v3-base-tasksource-nli",
        device=device,
        batch_size=128
    )

with open('test_shuffle_bis.txt', 'r') as file:
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

# print(classification_report(y_test, y_pred))
