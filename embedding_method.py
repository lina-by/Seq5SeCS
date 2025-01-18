from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import numpy as np
"""
# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`
query_prompt_name = "s2p_query"
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]
"""
query_prompt_name = "s2p_query"

with open('data/train.json', 'r') as file:
    train_file = json.load(file)
docs = []
line_to_label = {}
for category in train_file:
    for line in train_file[category]:
        docs.append(line)
        line_to_label[line] = category
        
with open('data/test_shuffle_bis.txt', 'r') as file:
    data = file.readlines()

data = [x.replace('\n', '') for x in data]
d = {'texts': data} 
test_df = pd.DataFrame(d)
test_df = test_df.dropna()
data = test_df["texts"].tolist()
queries = data
candidate_labels = ['Sports','Politics','Health','Finance','Travel','Food','Education','Environment','Fashion','Science','Technology','Entertainment']

        
# ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
doc_embeddings = model.encode(docs)
print(query_embeddings.shape, doc_embeddings.shape)
# (2, 1024) (2, 1024)

similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)

closest_sentences = np.argmax(similarities, axis=1)

labels = []
for i in range(len(closest_sentences)):
    #get label of this sentence based on train_json
    closest_sentence = closest_sentences[i]
    label = line_to_label[docs[closest_sentence]]
    labels.append({'labels': label, 'scores': similarities[i][closest_sentence]})
    
print(labels[:10])
    
    
y_pred = []
for i in range(len(labels)):
    label = labels[i]['labels']
    prob = labels[i]['scores']
    y_pred.append(label)
    
with open('y_pred_shuffle.txt', 'w') as file:
    for item in y_pred:
        file.write(item + "\n")