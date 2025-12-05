# Twitter Sentiment Analysis 

**Final Grade: 88%** 

This repository contains my completed coursework for the  Natural Language Processing module at QMUL.

The assignment focuses on building and optimizing a **sentiment classifier** for Twitter data, progressing from a simple baseline to an optimized production-ready model.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `NLP_Assignment_1.ipynb` | Baseline implementation (Q1-Q3) |
| `NLP_Q4.ipynb` | Optimized model implementation (Q4) |
| `NLP_REPORT.pdf` | 2-page technical report |
| `sentiment-dataset.tsv` | ~27,000 labeled tweets |
| `positive-words.txt` / `negative-words.txt` | Hu & Liu opinion lexicon |
| `error_analysis_q3.txt` / `error_analysis_q4.txt` | Detailed misclassification logs |

---

## 📊 Project Overview

**Task:** Binary sentiment classification (positive/negative) on ~27,000 tweets  
**Split:** 80% training (26,832 samples) / 20% test (6,708 samples)

---

## 🔧 Q1: Feature Extraction

Implemented a **bag-of-words** representation that converts tweets into feature dictionaries mapping tokens to frequency counts. A global dictionary tracks unique tokens across the corpus, enabling consistent feature vector construction for both training and test sets.

---

## 📈 Q2: Cross-Validation & Baseline

Established baseline performance using **10-fold cross-validation** with a **Linear SVM** classifier.

| Metric | Baseline Score |
|--------|----------------|
| Precision | 0.8290 |
| Recall | 0.8310 |
| **F1 Score** | **0.8294** |
| Accuracy | 0.8310 |

---

## 🔍 Q3: Error Analysis

Analyzed the confusion matrix to identify systematic failure patterns:

- **255 false positives** and **212 false negatives** in validation fold
- **Negation handling failures** — phrases like "not good" and "didn't have" were misclassified because tokens were treated independently
- **Sarcasm detection** — ironic hashtags like "#ThankYouObama" confused the model
- **Mixed-sentiment tweets** — posts containing both positive and negative signals
- **Context-dependent slang** — words like "bad" (meaning good) and "sick" (meaning awesome)

---

## ⚡ Q4: Model Optimization

Systematically improved both preprocessing and model architecture, achieving **+5.67% F1 improvement**.

### Preprocessing Improvements

| Technique | F1 Change | Notes |
|-----------|-----------|-------|
| Lowercase + word_tokenize | +2.25% | Reduces sparsity, captures contractions |
| Bigrams | +1.42% | Creates "not good" as distinct feature |
| Lemmatization | +1.24% | Consolidates morphological variants |
| Stopword removal | **-0.74%** | Lost sentiment signals ("but", "very") |
| **Combined best** | **+4.04%** | Near-additive improvements |

### Model Architecture

| Enhancement | F1 Change |
|-------------|-----------|
| Logistic Regression + TF-IDF | +0.86% |
| Hyperparameter tuning (C=5.0) | +0.08% |
| Hu & Liu opinion lexicon | +0.60% |
| Trigrams (1-3 n-grams) | +0.09% |

---

## 🏆 Final Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| F1 Score | 0.8294 | **0.8861** | +5.67% |
| Accuracy | 0.8310 | **0.8859** | +5.49% |

**Test Set Performance:** F1 = 0.8791, Accuracy = 0.8788  
*(Only 0.7% drop from cross-validation → minimal overfitting)*

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **scikit-learn** — LinearSVC, LogisticRegression, TF-IDF vectorization
- **NLTK** — tokenization, lemmatization, stopwords
- **NumPy** — numerical operations

---

## 🚀 Running the Notebooks

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install scikit-learn nltk numpy
   ```
3. Download NLTK data (runs automatically in notebook):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```
4. Place `sentiment-dataset.tsv` and lexicon files in the same directory
5. Run notebooks sequentially from top to bottom

---

## 💡 Key Insights

- **Twitter noise is signal** — removing @mentions, URLs, and hashtags *decreased* performance; these elements carry sentiment information
- **Stopwords matter for sentiment** — words like "but", "very", and "too" modify sentiment intensity
- **Bigrams solve negation** — the biggest single improvement came from capturing word pairs like "not good"
- **Domain knowledge helps** — opinion lexicons complement learned TF-IDF features

---

## 👤 Author

**Ahmed Idris**

