import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import ADASYN
import xgboost as xgb
import m2cgen as m2c
import json
import os

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

def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load and clean data
path = 'data/'
df = pd.read_csv(f'{path}twitter_parsed_dataset.csv').dropna()
df2 = pd.read_excel(f'{path}HateReviews.xlsx')
df3 = pd.read_excel(f'{path}modified_hate_speech_dataset.xlsx')
df = pd.concat([df, df2, df3], ignore_index=True)
df['Text'] = df['Text'].apply(clean)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X = vectorizer.fit_transform(df['Text'])
y = df['oh_label']

# Save vectorizer vocab + idf
vocab = vectorizer.vocabulary_
idf = vectorizer.idf_

os.makedirs("export", exist_ok=True)
vocab_clean = {str(k): int(v) for k, v in vocab.items()}
idf_clean = [float(val) for val in idf]

with open("export/vectorizer_data.json", "w") as f:
    json.dump({"vocab": vocab_clean, "idf": idf_clean}, f)

# Oversample
X_resampled, y_resampled = ADASYN().fit_resample(X, y) # type: ignore

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    base_score=0.5
)
model.fit(X_train, y_train)

# Export model to pure Python
code = m2c.export_to_python(model)
with open("export/model_code.py", "w") as f:
    f.write(code)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))