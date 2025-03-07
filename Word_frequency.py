from tqdm import tqdm
import string
import pandas as pd
import numpy as np
import re
from random import shuffle


class LanguageIdentification:
    def __init__(self, label, borne):
        self.index = {}
        self.stop_letters = (
            [str(x) for x in range(10)]
            + list(string.punctuation)
            + [" ", "“", "’", "”", "–", "‘"]
        )
        self.label_index = {}
        self.invert_index = {}
        for language in label:
            if language not in self.label_index:
                self.label_index[language] = len(self.label_index)
                self.invert_index[len(self.invert_index)] = language

        self.borne = borne

    def preprocess_word(self, word):
        for i, letter in enumerate(word):
            if letter in self.stop_letters:
                if i != len(word) - 1:
                    return self.preprocess_word(word[:i] + word[i + 1 :])
                else:
                    return word[:i]

        return word

    def train(self, text, label):
        for word in text.split():
            preprocessed_word = self.preprocess_word(word)
            if len(preprocessed_word) > 0 and len(preprocessed_word) <= self.borne:
                if preprocessed_word not in self.index:
                    self.index[preprocessed_word] = np.zeros(len(self.label_index))

                self.index[preprocessed_word][self.label_index[label]] += 1

    def end_training(self):
        words_to_delete = []
        for word in self.index:
            if np.sum(self.index[word]) <= 20:
                words_to_delete.append(word)
            else:
                self.index[word] /= np.sum(self.index[word])

        for word in words_to_delete:
            del self.index[word]

    def evaluate(self, text):
        score = np.zeros(len(self.label_index))

        for word in text.split():
            preprocessed_word = self.preprocess_word(word)
            if (
                len(preprocessed_word) > 0
                and len(preprocessed_word) <= self.borne
                and preprocessed_word in self.index
            ):
                score += self.index[preprocessed_word]
        return self.invert_index[np.argmax(score)]


def normalize_text(line):
    line = re.sub(r"'", " ' ", line)
    line = re.sub(r'"', "", line)
    line = re.sub(r"\.", " . ", line)
    line = re.sub(r"<br\s*/>", " ", line)
    line = re.sub(r",", " , ", line)
    line = re.sub(r"\(", " ( ", line)
    line = re.sub(r"\)", " ) ", line)
    line = re.sub(r"\!", " ! ", line)
    line = re.sub(r"\?", " ? ", line)
    line = re.sub(r"\;", " ", line)
    line = re.sub(r"\:", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def train_test_split(lines, labels):
    ids = list(range(len(lines)))
    shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]

    train_sentences = list(lines[train_ids])
    test_sentences = list(lines[test_ids])
    train_labels = list(labels[train_ids])
    test_labels = list(labels[test_ids])

    return train_sentences, test_sentences, train_labels, test_labels


def train_model(model, train_sentences, train_labels):
    print("################## BEGIN TRAINING ##################")
    for i, line in tqdm(enumerate(train_sentences)):
        line = normalize_text(line)
        model.train(line, train_labels[i])

    model.end_training()
    return model


def evaluate_model(model, test_sentences, test_labels):
    print("################## BEGIN EVALUATION ##################")
    labels_pred = []
    score = 0
    for i, line in tqdm(enumerate(test_sentences)):
        line = normalize_text(line)
        lang = model.evaluate(line)
        labels_pred.append(lang)
        if lang == test_labels[i]:
            score += 1

    return score / len(test_sentences), labels_pred


if __name__ == "__main__":
    data = pd.read_csv("data/train_submission.csv", encoding="utf-8")

    lines = data["Text"]
    labels = data["Label"]

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        lines, labels
    )

    model = LanguageIdentification(data["Label"], borne=7)

    model = train_model(model, train_sentences, train_labels)

    accuracy, labels_pred = evaluate_model(model, test_sentences, test_labels)

    print("Validation Accuracy: ", accuracy)
