# Seq5SeCS

Language Identification is an essential aspect of almost every Natural Language Processing (NLP) pipeline. However, this task appears to be extremely fastidious when it comes to rarer languages, for which obtaining a clear and unbiased dataset is particularly difficult. This work compares the performance and computational cost of different methods for language identification applied to a total of 390 languages.

## Method 1: Word frequency

TO DO

## Method 2: Inverse Word frequency

TO DO

## Method 3: Large Language Model

To train the XLM-RoBERTa model, run `python train_roberta.py` in `lm_training` folder. To infer the trained model on a dataset, run `python infer_roberta.py`.
