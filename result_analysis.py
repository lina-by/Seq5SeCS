import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tf_idf import separate_words

ROBERTA_LANGUAGES = json.load(open('roberta_languages.json'))

# Function to compute precision, recall, and F1, then plot them
def plots(y_pred, y_true, train_set):
    languages_data = {}

    # Compute sentence & word counts
    for lang, df in train_set.groupby('Label'):
        sentence_count = len(df)
        word_count = sum(len(separate_words(text)) for text in df.Text)

        languages_data[lang] = {'sentence_count': sentence_count, 'word_count': word_count}

    # Compute F1, Precision, and Recall
    labels = y_true['Label'].unique()
    f1 = f1_score(y_pred=y_pred['Label'], y_true=y_true['Label'], average=None)
    precision = precision_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels, average=None)
    recall = recall_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels, average=None)

    for i, lang in enumerate(labels):
        languages_data[lang]['f1_score'] = f1[i]
        languages_data[lang]['precision'] = precision[i]
        languages_data[lang]['recall'] = recall[i]

    # Create subplots: (1 row, 3 columns) to compare Words vs Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    for lang, data in languages_data.items():
        color = 'green' if lang in ROBERTA_LANGUAGES else 'blue'
        if 'f1_score' not in data:
            continue
        alpha = 0.5

        # Scatter plots with word count as x-axis
        ax1.scatter(data['word_count'], data['f1_score'], color=color, alpha=alpha)
        ax2.scatter(data['word_count'], data['precision'], color=color, alpha=alpha)
        ax3.scatter(data['word_count'], data['recall'], color=color, alpha=alpha)

    # Titles and Labels
    ax1.set_title("F1 Score vs Number of Words", fontsize=14)
    ax2.set_title("Precision vs Number of Words", fontsize=14)
    ax3.set_title("Recall vs Number of Words", fontsize=14)

    for ax in axes:
        ax.set_xlabel("Number of Words", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)

    # Simplified Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='RoBERTa-trained'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Other')]
    ax1.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()

# Function to compute precision, recall, and F1, then plot them
def plots_f1(y_pred, y_true, train_set):
    languages_data = {}

    # Compute sentence & word counts
    for lang, df in train_set.groupby('Label'):
        sentence_count = len(df)
        word_count = sum(len(separate_words(text)) for text in df.Text)

        languages_data[lang] = {'sentence_count': sentence_count, 'word_count': word_count}

    # Compute F1, Precision, and Recall
    labels = y_true['Label'].unique()
    f1 = f1_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels, average=None)
    precision = precision_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels, average=None)
    recall = recall_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels, average=None)

    for i, lang in enumerate(labels):
        languages_data[lang]['f1_score'] = f1[i]
        languages_data[lang]['precision'] = precision[i]
        languages_data[lang]['recall'] = recall[i]


    for lang, data in languages_data.items():
        color = 'green' if lang in ROBERTA_LANGUAGES else 'blue'
        if 'f1_score' not in data:
            continue
        alpha = 0.5

        # Scatter plots with word count as x-axis
        plt.scatter(data['word_count'], data['f1_score'], color=color, alpha=alpha)

    # Titles and Labels
    plt.title("F1 Score vs Number of Words", fontsize=14)
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # Simplified Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='RoBERTa-trained'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Other')]
    plt.legend(handles=handles, loc="lower right")

    plt.tight_layout()
    plt.show()

def plot_f1_vs_tokens_m(y_pred, y_true, train_set):
    
    
    f1 = f1_score(y_pred=y_pred['Label'], y_true=y_true['Label'], average=None)
    labels = y_true['Label'].unique()

    # Plot setup
    plt.figure(figsize=(10, 6))

    for i, lang in enumerate(labels):
        if lang in ROBERTA_LANGUAGES:
            plt.scatter(ROBERTA_LANGUAGES[lang]['tokens_m'], f1[i], color='green', alpha=0.5)
            plt.annotate(ROBERTA_LANGUAGES[lang]['language'], (ROBERTA_LANGUAGES[lang]['tokens_m'], f1[i]))

    # Titles and Labels
    plt.title("F1 Score vs Number of Tokens (Millions) in Original model", fontsize=14)
    plt.xlabel("Number of Tokens (M)", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)

    # Simplified Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='RoBERTa-trained')]
    plt.legend(handles=handles, loc="lower right")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def generate_language_info_table(y_pred, y_true, train_set):
    languages_data = {}

    # Compute sentence & word counts
    for lang, df in train_set.groupby('Label'):
        sentence_count = len(df)
        word_count = sum(len(separate_words(text)) for text in df.Text)
        languages_data[lang] = {'sentence_count': sentence_count, 'word_count': word_count}

    # Compute F1 scores
    labels = y_true['Label'].unique()
    f1 = f1_score(y_pred=y_pred['Label'], y_true=y_true['Label'], labels=labels , average=None)

    # Add F1 score to the language data
    for i, lang in enumerate(labels):
        languages_data[lang]['f1_score'] = f1[i]


    data = pd.DataFrame.from_dict(languages_data, orient='index')
    data['is_roberta'] = data.index.isin(ROBERTA_LANGUAGES.keys())

    def add_roberta_info(row):
        if row['is_roberta']:
            lang_info = ROBERTA_LANGUAGES[row.name]
            for key, value in lang_info.items():
                row[key] = value
        return row

    # Apply the function to add info for languages in the ROBERTA_LANGUAGES list
    data = data.apply(add_roberta_info, axis=1)

    return data


if __name__ == '__main__':
    train_set = pd.read_csv(r'data/train_submission.csv')
    y_true = pd.read_csv(r'data/y_true_test.csv')
    y_pred = pd.read_csv(r'data/y_pred_test.csv')
    
    plots_f1(y_pred=y_pred, y_true=y_true, train_set=train_set)
    plot_f1_vs_tokens_m(y_pred=y_pred, y_true=y_true, train_set=train_set)
