import json
import re
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from model_code import score

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

def clean(text):
    """
    Cleans text by converting to lowercase, removing URLs, mentions,
    punctuation, and stopwords.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)               # Remove mentions and standalone '#'
    text = re.sub(r'[^\w\s]', '', text)               # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

export_path = 'deprecated_model/model_export'
data_path = 'raw-data'

# Load the saved vectorizer data
with open(os.path.join(export_path, 'vectorizer_data.json'), "r") as f:
    vectorizer_data = json.load(f)

vectorizer = TfidfVectorizer(
    vocabulary=vectorizer_data["vocab"],
    ngram_range=(1, 3)
)
vectorizer.idf_ = np.array(vectorizer_data["idf"])

# Load test data and true labels
test_data = pd.read_excel('raw-data/modified_hate_speech_dataset.xlsx')
test_texts = test_data["Text"].values
y_true = test_data["oh_label"].values

print("Preprocessing test texts...")
processed_test_texts = [clean(text) for text in test_texts]

print("Transforming test texts into TF-IDF vectors...")
X_test_tfidf = vectorizer.transform(processed_test_texts).toarray()

print("Making predictions...")
# Get raw predictions, which are (N_samples, N_classes)
y_pred_raw = np.array([score(x) for x in X_test_tfidf])

# Convert (N_samples, N_classes) to (N_samples,) by taking the argmax for the predicted class
y_pred = np.argmax(y_pred_raw, axis=1)

print("\n--- Evaluation Results ---")
# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
