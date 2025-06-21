import json
import re
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from model_code import score # Assuming your model's prediction function is here

# Stopwords
stop_words = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are", "was", "were", 
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "so", "such", "too",
    "very", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
    "some", "no", "nor", "not", "only", "own", "same", "than", "can", "will", "just", "don", 
    "should", "now"
}

# Define the exact same clean function as used for training
def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)             # Remove mentions and standalone '#'
    text = re.sub(r'[^\w\s]', '', text)             # Remove punctuation (keeps alphanumeric and spaces)
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stop words
    return text

# 1. Load the saved vectorizer
# Using pathlib for cleaner path handling (optional but recommended)
from pathlib import Path
export_path = Path('/Users/marshall/Projects/REACHVERSE/Twitter Hate Speech/hate-speech-ml/export')
data_path = Path('/Users/marshall/Projects/REACHVERSE/Twitter Hate Speech/hate-speech-ml/data')

with open(export_path / 'vectorizer_data.json', "r") as f:
    vectorizer_data = json.load(f)

vectorizer = TfidfVectorizer(
    vocabulary=vectorizer_data["vocab"],
    ngram_range=(1, 3)
)
vectorizer.idf_ = np.array(vectorizer_data["idf"])

# 2. Load test data (text and true labels)
test_data = pd.read_excel(data_path / 'modified_hate_speech_dataset.xlsx')
test_texts = test_data["Text"].values
y_true = test_data["oh_label"].values

# 3. Preprocess text - APPLY THE SAME CLEAN FUNCTION!
print("Preprocessing test texts...")
processed_test_texts = [clean(text) for text in test_texts]

print("Transforming test texts into TF-IDF vectors...")
X_test_tfidf = vectorizer.transform(processed_test_texts).toarray()

# 4. Predict
print("Making predictions...")
# Get raw predictions, which are (N_samples, N_classes)
y_pred_raw = np.array([score(x) for x in X_test_tfidf])

#Convert (N_samples, 2) to (N_samples,) by taking argmax
y_pred = np.argmax(y_pred_raw, axis=1)

# 5. Calculate F1 score
f1 = f1_score(y_true, y_pred) # This is where the error occurs
print("\n--- Evaluation Results ---")
print("F1 Score:", f1)

# 6. Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))