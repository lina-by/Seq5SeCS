import os
import pandas as pd
import numpy as np
import re
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def load_and_prepare_data(filepath):
    train_set = pd.read_csv(filepath).dropna()
    train_s, validation_s = train_test_split(train_set, test_size=0.2, random_state=42)
    
    corpus = []
    for lang, df in train_s.groupby('Label'):
        doc = []
        for phrase in df.Text:
            doc += tokenize(phrase)
        doc = ' '.join(doc)
        corpus.append({'langue': lang, 'text': doc})
    
    corpus = pd.DataFrame(corpus)
    return train_s, validation_s, corpus

def perform_tfidf(corpus, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus.text)
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(output_dir, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(X, f)
    return X

def perform_pca(X, output_dir, n_components=388):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=n_components, svd_solver='full')
    X_pca = pca.fit_transform(X.toarray())
    with open(os.path.join(output_dir, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(output_dir, 'pca_matrix.pkl'), 'wb') as f:
        pickle.dump(X_pca, f)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance (Original Params: {X.shape[1]})')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'pca_variance_plot.png'))
    plt.close()
    return pca, X_pca

def retrieve(x, X_pca, corpus):
    x = x.reshape(1, -1)
    similarities = cosine_similarity(X_pca, x)
    most_similar_index = np.argmax(similarities)
    return corpus.langue.iloc[most_similar_index]

def evaluate_model(validation_s, vectorizer, pca, X_pca, corpus, n_components=None, X_test_pca=None):
    if X_test_pca is None:
        X_test = vectorizer.transform(validation_s.Text)
        X_test_pca = pca.transform(X_test.toarray())
    if n_components is not None:
        X_pca = X_pca[:, :n_components]
        X_test_pca = X_test_pca[:, :n_components]
    y_predict = np.apply_along_axis(retrieve, 1, X_test_pca, X_pca, corpus)
    accuracy = accuracy_score(validation_s.Label, y_predict)
    return y_predict, accuracy

def analyze_performance(validation_s, vectorizer, pca, X_pca, output_dir, X_test_pca=None):
    max_iter = X_pca.shape[1]
    components = np.arange(int(max_iter/2), max_iter, 5)
    accuracies = []
    variance_ratios = []
    
    for k in tqdm(components):
        _, accuracy = evaluate_model(validation_s, vectorizer, pca, X_pca, corpus, n_components=k, X_test_pca=X_test_pca)
        accuracies.append(accuracy)
        variance_ratios.append(np.sum(pca.explained_variance_ratio_[:k]))
    
    plot_performance(components, accuracies, variance_ratios, output_dir)

def plot_performance(components, accuracies, variance_ratios, output_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(components, accuracies, marker='s', linestyle='-', color='b', label='Accuracy')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Accuracy', color='b')
    ax2 = ax1.twinx()
    ax2.plot(components, variance_ratios, marker='o', linestyle='--', color='r', label='Variance')
    ax2.set_ylabel('Cumulative Explained Variance', color='r')
    plt.title('Accuracy and Variance vs. Number of Components')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'pca_tf_idf_plot.png'))
    plt.show()

def visualize_errors(validation_s, y_predict, corpus, output_dir):
    word_counts = corpus.set_index('langue').text.apply(lambda x: len(x.split()))
    labels = set(validation_s.Label.unique()) & set(corpus.langue.unique())
    tpr = {}
    tnr = {}
    
    for label in labels:
        true_positives = np.sum((validation_s.Label == label) & (y_predict == label))
        false_negatives = np.sum((validation_s.Label == label) & (y_predict != label))
        tpr[label] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    
    plt.figure(figsize=(10, 6))
    common_labels = list(tpr.keys())
    plt.scatter(word_counts.loc[common_labels], [tpr[l] for l in common_labels], marker='o', linestyle='-', color='b', label='True Positive Rate')
    plt.xlabel('Number of Words in Corpus per Language')
    plt.ylabel('Rate')
    plt.title('True Positive and vs. Corpus Size per Language')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'error_visualization.png'))
    plt.close()

if __name__ == "__main__":
    data_path = 'data/train_submission.csv'
    output_dir = 'tf_idf'
    
    if not os.path.exists(output_dir):
        print('compute pca')
        train_s, validation_s, corpus = load_and_prepare_data(data_path)
        X = perform_tfidf(corpus, output_dir)
        pca, X_pca = perform_pca(X, output_dir)
    
    train_s, validation_s, corpus = load_and_prepare_data(data_path)
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(output_dir, 'tfidf_matrix.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(output_dir, 'pca_model.pkl'), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(output_dir, 'pca_matrix.pkl'), 'rb') as f:
        X_pca = pickle.load(f)
    
    y_predict, accuracy = evaluate_model(validation_s, vectorizer, pca, X_pca, corpus)
    print(f'Final Accuracy: {accuracy * 100:.2f}%')
    
    visualize_errors(validation_s, y_predict, corpus, output_dir)
    
    # X_test = vectorizer.transform(validation_s.Text)
    # X_test_pca = pca.transform(X_test.toarray())
    # analyze_performance(validation_s, vectorizer, pca, X_pca, output_dir, X_test_pca=X_test_pca)
