# **🧠 Tweet Sentiment Analysis with TF-IDF and ML Models**

This project focuses on performing sentiment analysis on Twitter data using traditional machine learning models and TF-IDF feature extraction.  
We evaluate three popular classifiers — **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM** — to determine which performs best on classifying tweets as **positive or negative**.

---

## 📁 Dataset

The dataset used is [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, containing **1.6 million tweets** labeled as:

-   `0`: Negative
-   `4`: Positive

> Note: The dataset originally includes a `2` (neutral) class, but no actual examples were present. The task was converted into a **binary classification** problem:  
> `0 → Negative`, `4 → Positive` → **remapped as 0 and 1**.

---

## ⚙️ Preprocessing Pipeline

To clean and standardize tweet texts, we applied the following preprocessing steps:

-   🔡 **Lowercasing**
-   🔎 **Contraction Expansion** (`can't → cannot`)
-   🔗 **Remove URLs, Mentions, Emojis**
-   ✂️ **Tokenization**
-   🧹 **Stopword Removal**
-   🚫 **Negation Handling** (`not good → not_good`)
-   🧬 **Lemmatization** (`running → run`)

Empty tweets after preprocessing were removed from the dataset.

---

## 🧮 Feature Extraction

Text data was transformed into numerical vectors using **TF-IDF** (Term Frequency – Inverse Document Frequency):

```python
TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
```

-   Considers both **unigrams** and **bigrams**
-   Filters out overly common or rare words
-   Captures the importance of words across the corpus

---

## 🤖 Models Trained

| Model                   | Description                                      |
| ----------------------- | ------------------------------------------------ |
| **Naive Bayes**         | Fast baseline classifier for text data           |
| **Logistic Regression** | Linear model that performs very well with TF-IDF |
| **Linear SVM**          | Maximizes the margin between sentiment classes   |

---

## 📊 Evaluation Metrics

All models were evaluated using:

-   ✅ **Accuracy**
-   🎯 **F1 Score**
-   🎯 **Precision**
-   🎯 **Recall**
-   📉 **Confusion Matrix**

| Model               | Accuracy   | F1 Score   | Precision | Recall     |
| ------------------- | ---------- | ---------- | --------- | ---------- |
| Naive Bayes         | 0.7602     | 0.7594     | 0.7616    | 0.7572     |
| Logistic Regression | **0.7764** | **0.7811** | 0.7647    | 0.7982     |
| Linear SVM          | 0.7752     | 0.7810     | 0.7611    | **0.8019** |

> 🔎 **Logistic Regression** was selected as the best model due to its balance across all metrics.

---

## 📈 Visualizations

-   📊 Token count before vs. after preprocessing
-   ☁️ WordClouds of raw and cleaned tweets
-   📉 Confusion matrices for all models
-   📊 Model comparison bar charts across evaluation metrics

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook:

    ```bash
    jupyter notebook sentiment_analysis.ipynb
    ```

---

## 🧩 Future Work

-   Deep learning models (LSTM, BERT)
-   Sentiment trend analysis over time
-   Emoji/hashtag semantic enhancement

---

## 📬 Contact

Developed by [Mert Çetin](https://github.com/your-username)
📧 Feel free to reach out for collaborations or feedback!
